#!/usr/bin/env python3
"""Run the independent review of a listar2000/miles PR with a fresh agent, as SKILL.md "Supervised runs" says."""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_SLUG = "listar2000/miles"
ATTRIBUTION = re.compile(
    r"requested by|claude-projects-attribution|generated with \[?claude|claude\.ai/|claude-session",
    re.I,
)
SEVERITY_ORDER = ("blocking", "should-fix", "nit")
SCHEMA = HERE / "output.schema.json"
PROMPT = """Review pull request #{pr} of listar2000/miles (head {head}, merge base {base}) as the
independent reviewer. Your guidance is .claude/skills/miles-review/SKILL.md, its "Supervised runs"
section included. Everything you need is below: the skill, the rules it points to, the bundle and
every changed file at the head. Do not run commands. REVIEW.md (upstream's review guide) is left out
on purpose; .agents/rules is the same as .claude/rules, included when the PR touches miles/. Answer
with the JSON object the skill describes.
"""
FOLLOW_UP = """Round {round} of your review of PR #{pr}: the head moved from {prior} to {head} (merge base
{base} unchanged). Below are the diff since {prior}, the files it changed at the new head, and the refreshed
bundle (checks, commits, PR body). Re-check each finding of your last answer and anything the new diff adds,
then answer with the same JSON for the whole PR at {head}, its `prior` listing each of your last findings.
"""
DEEP = """This is a deep review: do section 5 of the skill as well. The `head:` files include the
--context files, and `evidence/` holds the logs and results behind the PR's numbers.
"""
MAX_EVIDENCE_CHARS = 200_000  # per evidence file; the head and the tail are kept
CONTEXT_FILES = (
    ".claude/skills/miles-review/SKILL.md",
    "fork/README.md",
    "fork/rfcs/0001-main-and-stable-versions.md",
    ".claude/skills/miles-fork/SKILL.md",
    ".github/pull_request_template.md",
)
MAX_PROMPT_CHARS = 900_000  # Codex rejects prompts over 1,048,576 characters
TEST_FILE = re.compile(r"(^|/)(tests?/|test_[^/]*\.py$)")
FIXUP = re.compile(r"^(fixup|amend|squash)! ", re.M)
MARKER = (
    "<!-- miles-review head={head} mode={mode} verdict={verdict} round={round} session={session} rules={rules} -->"
)
ROUND = re.compile(
    r"<!-- miles-review head=(?P<head>\w+) mode=(?P<mode>\w+) verdict=[\w-]+(?: round=(?P<round>\d+))?"
    r"(?: session=(?P<session>[\w-]+) rules=(?P<rules>\w+))? -->"
)
LINE_MARK = "<!-- miles-review-line -->"
THIS_ROUND = re.compile(r"<!-- this-round: (.*) -->")
CHECK_IMAGE = "docker.io/starlimobius/miles-mobius:main"
STATUS = "codex-review"  # the commit status a branch rule on main can require
EVIDENCE_VOLUMES = re.compile(r"(miles|mdev|rl)-[\w.-]+")  # ours; never another team's volume


def codex(args, out, session=None):
    """A saved session (no --ephemeral), so the next round can resume it; `resume` takes the sandbox as config."""
    return [
        "codex",
        "exec",
        *(["resume"] if session else []),
        "-m",
        args.model,
        "-c",
        f"model_reasoning_effort={args.effort}",
        "-c",
        'sandbox_mode="read-only"',
        "--skip-git-repo-check",
        "--json",
        "--output-schema",
        str(SCHEMA),
        "-o",
        str(out),
        *([session] if session else []),
        "-",
    ]


# Each adapter returns the command that runs one agent non-interactively and read-only, with the prompt on stdin
# and its final message written to `out`; with `session`, it continues that saved session instead.
AGENTS = {"codex": codex}
DEFAULT_MODELS = {"codex": "gpt-6-astra"}
VERDICT_FOR = {"blocking": "blocked", "should-fix": "ready after the should-fixes", None: "ready"}


def sh(cmd, cwd=None, timeout=None, check=True, input=None, env=None):
    p = subprocess.run(
        cmd,
        cwd=cwd,
        env={**os.environ, **env} if env else None,
        timeout=timeout,
        text=True,
        capture_output=True,
        input=input,
        stdin=None if input is not None else subprocess.DEVNULL,
        shell=isinstance(cmd, str),
    )
    if check and p.returncode:
        sys.exit(f"command failed ({p.returncode}): {cmd}\n{p.stdout[-2000:]}{p.stderr[-2000:]}")
    return p


@contextlib.contextmanager
def repo_lock(repo):
    """Serialise git writes to the reviewer's checkout, which parallel reviews, the daemon's update and /merge share."""
    git_dir = sh(["git", "rev-parse", "--git-common-dir"], cwd=repo).stdout.strip()
    with open(Path(repo) / git_dir / "miles-review.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def tail(p, n=25):
    lines = (p.stdout + p.stderr).strip().splitlines()
    return "\n".join(lines[-n:])


class LocalRunner:
    """Runs the PR's own code (ledger, pre-commit, tests) in the review checkout on this machine."""

    def __init__(self, tree, python):
        self.tree, self.python, self.precommit = tree, python, ["uvx", "pre-commit"]

    def run(self, cmd, timeout=None, check=False):
        return sh(cmd, cwd=self.tree, timeout=timeout, check=check)

    def close(self):
        pass


class ModalRunner:
    """Runs PR checks in a CPU Modal sandbox, away from this machine's credentials."""

    def __init__(self, tree, revs, work):
        import modal

        bundle = work / "pr.bundle"
        for i, rev in enumerate(revs):
            sh(["git", "update-ref", f"refs/review/rev{i}", rev], cwd=tree)
        sh(
            [
                "git",
                "bundle",
                "create",
                "-q",
                str(bundle),
                *(f"refs/review/rev{i}" for i in range(len(revs))),
                "--tags",
            ],
            cwd=tree,
        )
        image = modal.Image.from_registry(CHECK_IMAGE).add_local_file(bundle, "/tmp/pr.bundle")
        app = modal.App.lookup("miles-review-checks", create_if_missing=True)
        self.sb = modal.Sandbox.create("sleep", "infinity", image=image, app=app, cpu=4, memory=8192, timeout=3600)
        self.tree, self.python, self.precommit = "/root/pr", "python", ["python", "-m", "pre_commit"]
        self._exec(
            f"git init -q {self.tree} && git -C {self.tree} fetch -q /tmp/pr.bundle 'refs/review/*:refs/review/*' 'refs/tags/*:refs/tags/*' && git -C {self.tree} checkout -q {revs[0]}",
            600,
            True,
        )
        self._exec("pip install -q pre-commit", 600, True)

    def _exec(self, script, timeout, check):
        p = self.sb.exec(
            "bash", "-c", f"cd {self.tree} 2>/dev/null; export PYTHONPATH={self.tree}; {script}", timeout=timeout
        )
        p.wait()
        out = subprocess.CompletedProcess(script, p.returncode, p.stdout.read(), p.stderr.read())
        if check and out.returncode:
            sys.exit(f"sandbox command failed ({out.returncode}): {script}\n{out.stdout[-2000:]}{out.stderr[-2000:]}")
        return out

    def run(self, cmd, timeout=None, check=False):
        return self._exec(" ".join(shlex.quote(c) for c in cmd), timeout or 1800, check)

    def close(self):
        self.sb.terminate()


def prepare(args, work):
    meta = json.loads(
        sh(
            [
                "gh",
                "pr",
                "view",
                str(args.pr),
                "-R",
                REPO_SLUG,
                "--json",
                "number,title,body,headRefOid,headRefName,isDraft,url",
            ]
        ).stdout
    )
    if args.body:
        meta["body"] = args.body.read_text()
    with repo_lock(args.repo):
        sh(["git", "fetch", "-q", "--tags", "origin", "main"], cwd=args.repo)
        if args.head:
            sh(["git", "update-ref", f"refs/review/pr{args.pr}", args.head], cwd=args.repo)
        else:
            sh(["git", "fetch", "-q", "origin", f"+pull/{args.pr}/head:refs/review/pr{args.pr}"], cwd=args.repo)
        head = meta["headRefOid"] = sh(["git", "rev-parse", f"refs/review/pr{args.pr}"], cwd=args.repo).stdout.strip()
        tree = work / "tree"
        if tree.exists():
            shutil.rmtree(tree)
        sh(["git", "clone", "-q", "--local", "--no-checkout", str(args.repo), str(tree)])
        sh(
            [
                "git",
                "fetch",
                "-q",
                str(args.repo),
                "refs/remotes/origin/main:refs/remotes/origin/main",
                f"refs/review/pr{args.pr}:refs/review/pr{args.pr}",
            ],
            cwd=tree,
        )
        sh(["git", "checkout", "-q", head], cwd=tree)
        base_ref = "origin/main"
        if stacked := re.match(r"\s*Stacked on #(\d+)", meta["body"] or ""):
            n = stacked.group(1)
            sh(["git", "fetch", "-q", "origin", f"+pull/{n}/head:refs/review/pr{n}"], cwd=args.repo)
            sh(["git", "fetch", "-q", str(args.repo), f"refs/review/pr{n}:refs/review/pr{n}"], cwd=tree)
            base_ref = f"refs/review/pr{n}"
    base = sh(["git", "merge-base", base_ref, "HEAD"], cwd=tree).stdout.strip()
    meta["folded"] = fold_fixups(tree, base)
    return meta, sh(["git", "rev-parse", "HEAD"], cwd=tree).stdout.strip(), base, tree


def fold_fixups(tree, base):
    """Fold fixup!/amend! commits as the merge desk will, so commits and checks are judged as they land."""
    if not FIXUP.search(sh(["git", "log", "--format=%s", f"{base}..HEAD"], cwd=tree).stdout):
        return None
    p = sh(
        [
            "git",
            "-c",
            "user.name=miles-review",
            "-c",
            "user.email=review@localhost",
            "rebase",
            "-q",
            "-i",
            "--autosquash",
            base,
        ],
        cwd=tree,
        check=False,
        env={"GIT_SEQUENCE_EDITOR": ":"},
    )
    if p.returncode:
        sh(["git", "rebase", "--abort"], cwd=tree, check=False)
        return "fixups do not fold cleanly; reviewed unfolded"
    return f"fixups folded onto {base[:8]} as {sh(['git', 'rev-parse', 'HEAD'], cwd=tree).stdout.strip()[:8]}"


def run_checks(runner, rev, files):
    runner.run(["git", "checkout", "-q", rev], check=True)
    out = {}
    out["ledger"] = tail(runner.run([runner.python, "fork/tools/ledger.py", "check"]), 5)
    listed = runner.run(["git", "ls-tree", "-r", "--name-only", rev, "--", *files]).stdout.split() if files else []
    if listed:
        p = runner.run([*runner.precommit, "run", "--files", *listed], timeout=1800)
        kept = [
            ln for ln in tail(p, 400).splitlines() if not ln.endswith(("Passed", "Skipped")) and not ln.startswith("[")
        ]
        out["precommit"] = "\n".join(kept + [f"exit {p.returncode}" + ("" if p.returncode else ": all hooks passed")])
        runner.run(["git", "checkout", "-q", "--", "."])
    else:
        out["precommit"] = "(no changed files at this revision)"
    pytest = [runner.python, "-m", "pytest", "miles_ext/tests", "-q", "-p", "no:cacheprovider"]
    out["pytest"] = tail(runner.run([*pytest, "--continue-on-collection-errors"], timeout=1800), 15)
    runner.run(["git", "clean", "-fdq"])
    return out


def added_lines(diff):
    added, current, n = {}, "", 0
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
        elif m := re.match(r"@@ -\S+ \+(\d+)", line):
            n = int(m.group(1))
        elif line.startswith("+"):
            added.setdefault(current, set()).add(n)
            n += 1
        elif not line.startswith("-"):
            n += 1
    return added


def comment_blocks(tree, added):
    found = []
    for f, lines in added.items():
        path = tree / f
        if not f.endswith(".py") or not path.exists():
            continue
        start, n = 0, 0
        for i, line in enumerate(path.read_text(errors="replace").splitlines() + [""], 1):
            if line.lstrip().startswith("#"):
                start, n = (start or i), n + 1
                continue
            if n >= 3 and lines & set(range(start, i)):
                found.append(f"{f}:{start}: {n}-line comment block")
            start, n = 0, 0
    return found


def ratios(diff):
    test = prod = comment = nonblank = 0
    current = ""
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
        elif line.startswith("+") and not line.startswith("+++") and current.endswith(".py"):
            body = line[1:].strip()
            if not body:
                continue
            nonblank += 1
            comment += body.startswith("#")
            if "/tests/" in f"/{current}" or Path(current).name.startswith("test_"):
                test += 1
            else:
                prod += 1
    flags = []
    if prod and test > 1.5 * prod:
        flags.append("test lines exceed 1.5x production lines")
    if nonblank and comment > 0.15 * nonblank:
        flags.append("comments exceed 15% of added lines")
    return (
        f"added python lines: production {prod}, test {test}, comment {comment} of {nonblank} non-blank\n"
        f"flags: {', '.join(flags) or 'none'}"
    )


def new_tests(diff):
    nodes, current = [], ""
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
        m = re.match(r"\+\s*(?:async\s+)?def (test_\w+)", line)
        if m and current.endswith(".py"):
            nodes.append(f"{current}::{m.group(1)}")
    return nodes


def pytest_node(runner, node):
    t0 = time.time()
    p = runner.run([runner.python, "-m", "pytest", node, "-q", "-p", "no:cacheprovider"], timeout=600)
    result = {0: "pass", 1: "fail"}.get(p.returncode, "error")
    return result, round(time.time() - t0, 1), tail(p, 3)


def prove_tests(runner, base, files, nodes):
    """Run each new test on the head, then with the base's version of every changed production file."""
    proofs = {node: {"head": pytest_node(runner, node)} for node in nodes}
    for f in files:
        if TEST_FILE.search(f):
            continue
        if runner.run(["git", "cat-file", "-e", f"{base}:{f}"]).returncode == 0:
            runner.run(["git", "checkout", "-q", base, "--", f], check=True)
        else:
            runner.run(["rm", "-f", f])
    for node in nodes:
        proofs[node]["base"] = pytest_node(runner, node)
    runner.run(["git", "reset", "-q", "--hard"])
    runner.run(["git", "clean", "-fdq"])
    return proofs


def build_bundle(args, work, meta, head, base, tree):
    bundle = work / "bundle"
    bundle.mkdir(exist_ok=True)
    git = lambda *a: sh(["git", *a], cwd=tree).stdout  # noqa: E731
    diff = git("diff", base, head)
    files = git("diff", "--name-only", base, head).split()
    commits = git("log", "--format=%h %s%n%b%n%(trailers:only)---", f"{base}..{head}")
    (bundle / "pr.json").write_text(json.dumps(meta, indent=2))
    (bundle / "commits.txt").write_text((f"({meta['folded']})\n" if meta["folded"] else "") + commits)
    (bundle / "diff.patch").write_text(diff)
    (bundle / "diffstat.txt").write_text(git("diff", "--stat", base, head))
    (bundle / "files.txt").write_text("\n".join(files) + "\n")
    nodes = new_tests(diff)
    # The checks depend on the two trees, the commit messages (the ledger reads trailers) and this script, so a
    # round whose code and commits did not move (a PR body fix) reuses them.
    trees = sh(["git", "rev-parse", f"{head}^{{tree}}", f"{base}^{{tree}}"], cwd=tree).stdout.split()
    messages = git("log", "--format=%B", f"{base}..{head}")  # not the shas, which folding renews each round
    where = [args.checks_on, CHECK_IMAGE if args.checks_on == "modal" else args.python]
    key = hashlib.sha256("\n".join([*trees, messages, *where, Path(__file__).read_text()]).encode()).hexdigest()[:16]
    cache = args.out / "checks-cache" / f"{key}.json"
    if cached := cache.exists():
        ran = json.loads(cache.read_text())
    else:
        runner = ModalRunner(tree, (head, base), work) if args.checks_on == "modal" else LocalRunner(tree, args.python)
        try:  # a stopped review (SIGTERM) still ends its sandbox
            ran = {"ledger_list": runner.run([runner.python, "fork/tools/ledger.py", "list"], check=True).stdout}
            ran["checks"] = {"head": run_checks(runner, head, files), "base": run_checks(runner, base, files)}
            runner.run(["git", "checkout", "-q", head], check=True)
            ran["proofs"] = prove_tests(runner, base, files, nodes)
        finally:
            runner.close()
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(ran))
    checks, proofs = ran["checks"], ran["proofs"]
    (bundle / "ledger_list.txt").write_text(ran["ledger_list"])
    with repo_lock(args.repo):
        sh(["git", "fetch", "-q", "origin", "+refs/heads/*:refs/remotes/origin/*"], cwd=args.repo, check=False)
    taken = sh(
        [
            "git",
            "log",
            "--remotes=origin",
            "--format=%D %(trailers:key=Fork-Fix,key=Fork-Hook,key=Fork-Ext,valueonly,separator=%x20)",
        ],
        cwd=args.repo,
    ).stdout
    ids = sorted({i for i in re.findall(r"\b(?:fix|hook|ext)-\d{3}\b", taken)})
    (bundle / "ids_on_branches.txt").write_text(
        "ids in commits on any branch of origin, merged or not:\n" + " ".join(ids)
    )
    if args.board:
        reserved = [line for line in args.board.read_text().splitlines() if re.search(r"reserv", line, re.I)]
        (bundle / "reserved_ids.txt").write_text("\n".join(reserved) or "none")
    for name in ("ledger", "precommit", "pytest"):
        (bundle / f"{name}.txt").write_text(
            f"== head {head[:8]}\n{checks['head'][name]}\n\n== base {base[:8]}\n{checks['base'][name]}\n"
        )
    (bundle / "test_proofs.txt").write_text(
        "\n".join(
            f"{n}: head {p['head'][0]} ({p['head'][1]} s), base {p['base'][0]} ({p['base'][1]} s)\n"
            f"  head: {p['head'][2]}\n  base: {p['base'][2]}"
            for n, p in proofs.items()
        )
        or "no new tests"
    )
    (bundle / "comment_blocks.txt").write_text("\n".join(comment_blocks(tree, added_lines(diff))) or "none")
    (bundle / "ratios.txt").write_text(ratios(diff))
    hits = [f"PR body: {line}" for line in (meta["body"] or "").splitlines() if ATTRIBUTION.search(line)]
    hits += [f"commits: {line}" for line in commits.splitlines() if ATTRIBUTION.search(line)]
    (bundle / "attribution.txt").write_text("\n".join(hits) or "none")
    if args.prior:
        shutil.copy(args.prior, bundle / "prior_review.md")
    return bundle, {"files": files, "checks": checks, "attribution": hits, "proofs": proofs, "cached": cached}


def build_prompt(args, tree, head, base, bundle, files):
    rules = sorted((tree / ".claude/rules").glob("*.md")) if any(f.startswith("miles/") for f in files) else []
    # The skill comes from the supervisor's own copy, so a PR that edits it is still reviewed under the current one.
    sections = [
        (p, ((HERE / "SKILL.md") if p.endswith("miles-review/SKILL.md") else tree / p).read_text())
        for p in CONTEXT_FILES
    ]
    sections += [(str(r.relative_to(tree)), r.read_text()) for r in rules]
    sections += [(f"bundle/{b.name}", b.read_text(errors="replace")) for b in sorted(bundle.iterdir())]
    head_files = [(f"head:{f}", (tree / f).read_text(errors="replace")) for f in files if (tree / f).is_file()]
    sections += deep_inputs(args, tree, files)  # no stand-in in the diff, so they are never dropped
    body = "\n".join(f"\n===== {name} =====\n{text}" for name, text in sections + head_files)
    if len(body) > MAX_PROMPT_CHARS:  # too big for the agent: the diff in bundle/ has to stand in for full files
        body = "\n".join(f"\n===== {name} =====\n{text}" for name, text in sections)
    if len(body) > MAX_PROMPT_CHARS:
        sys.exit(
            f"the bundle{', context and evidence' if args.deep else ''} alone are {len(body)} characters, over "
            f"{MAX_PROMPT_CHARS}; narrow them, split the PR or review it by hand"
        )
    return PROMPT.format(pr=args.pr, head=head, base=base) + (DEEP if args.deep else "") + body


def rules_hash(tree):
    """What a resumed session was told (the skill, the rules the prompt inlines, the prompts); when any of it
    changes, the next round starts a fresh session."""
    docs = [
        HERE / "SKILL.md",
        SCHEMA,
        *(tree / f for f in CONTEXT_FILES[1:]),
        *sorted((tree / ".claude/rules").glob("*.md")),
    ]
    text = b"".join(d.read_bytes() for d in docs if d.is_file()) + (PROMPT + FOLLOW_UP + DEEP).encode()
    return hashlib.sha256(text).hexdigest()[:12]


def follow_up(args, tree, meta, base, bundle, last):
    """The prompt that continues the last round's session with what changed since; None for a fresh round."""
    if args.fresh or not last or last["session"] in (None, "none") or last["rules"] != meta["rules"]:
        return None
    if last["mode"] != ("deep" if args.deep else "normal"):
        return None
    prior, head = last["head"], meta["headRefOid"]
    if sh(["git", "cat-file", "-e", f"{prior}^{{commit}}"], cwd=tree, check=False).returncode:
        url = sh(["git", "remote", "get-url", "origin"], cwd=args.repo).stdout.strip()
        sh(["git", "fetch", "-q", url, prior], cwd=tree, check=False)
    if sh(["git", "merge-base", "--is-ancestor", base, prior], cwd=tree, check=False).returncode:
        return None  # rebased (or the old head is gone): the diff since would carry main's changes
    delta = sh(["git", "diff", prior, head], cwd=tree).stdout
    changed = sh(["git", "diff", "--name-only", prior, head], cwd=tree).stdout.split()
    sections = [("delta.patch", delta or "(no code change)")]
    sections += [(f"head:{f}", (tree / f).read_text(errors="replace")) for f in changed if (tree / f).is_file()]
    sections += [(f"bundle/{b.name}", b.read_text(errors="replace")) for b in sorted(bundle.iterdir())
                 if b.name not in ("diff.patch", "prior_review.md")]  # fmt: skip
    if args.deep:  # context and evidence the session already read are not sent again
        seen = seen_deep(args, last["session"])
        sections += [(n, t) for n, t in deep_inputs(args, tree, changed) if seen.get(n) != digest(t)]
    body = "\n".join(f"\n===== {name} =====\n{text}" for name, text in sections)
    if len(body) > MAX_PROMPT_CHARS:
        return None
    rnd = int(last["round"] or 1) + 1
    return FOLLOW_UP.format(round=rnd, pr=args.pr, prior=prior[:8], head=head[:8], base=base[:8]) + body


def deep_inputs(args, tree, files):
    """A deep review's --context files (minus `files`, inlined already) and --evidence, as (name, text)."""
    if not args.deep:
        return []
    found = [(f"head:{f}", (tree / f).read_text(errors="replace")) for f in context_files(args, tree, files)]
    return found + [(f"evidence/{name}", text) for name, text in evidence(args)]


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def seen_deep(args, session):
    """Deep inputs `session` has read, as {name: digest}; empty when the record is another session's."""
    f = args.out / f"pr{args.pr}-deep-seen.json"
    record = json.loads(f.read_text()) if f.exists() else {}
    return record.get("seen", {}) if record.get("session") == session else {}


def set_status(meta, state, text, url=None):
    """The PR head's `codex-review` commit status; best effort (the token needs Commit statuses: write)."""
    try:
        gh_api("-X", "POST", f"repos/{REPO_SLUG}/statuses/{meta['headRefOid']}", "-f", f"state={state}",
               "-f", f"context={STATUS}", "-f", f"description={text[:140]}", "-f", f"target_url={url or meta['url']}")  # fmt: skip
    except SystemExit as e:
        print(f"could not set the {STATUS} status: {e}", file=sys.stderr)


def context_files(args, tree, files):
    """The --context paths at the head (globs allowed), minus the changed files already inlined."""
    inside = lambda p: p.is_file() and p.resolve().is_relative_to(tree.resolve())  # noqa: E731  no ../ or symlinks out
    found = sorted({str(p.relative_to(tree)) for g in args.context for p in tree.glob(g) if inside(p)} - set(files))
    missing = [g for g in args.context if not any(tree.glob(g))]
    if missing:
        sys.exit(f"--context matches nothing at the head: {', '.join(missing)}")
    return found


def evidence(args):
    """(name, text) for each --evidence file, or each text file under an --evidence directory."""
    if not args.evidence:
        return []
    roots = [Path(e).resolve() for e in args.evidence]
    top = Path(os.path.commonpath([r if r.is_dir() else r.parent for r in roots]))  # names stay distinct
    out = []
    for root in roots:
        paths = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]
        for p in paths:
            text = p.read_text(errors="replace")
            if "\0" in text[:4096]:
                continue  # binary
            if len(text) > MAX_EVIDENCE_CHARS:
                half = MAX_EVIDENCE_CHARS // 2
                text = f"{text[:half]}\n[... {len(text) - 2 * half} characters cut ...]\n{text[-half:]}"
            out.append((str(p.relative_to(top)), text))
    return out


def fetch_evidence(items, work):
    """Local copies of ``modal://<volume>/<path>`` evidence (our volumes only); local paths pass through."""
    out = []
    for item in items:
        if not item.startswith("modal://"):
            out.append(item)
            continue
        volume, _, path = item.removeprefix("modal://").partition("/")
        if not EVIDENCE_VOLUMES.fullmatch(volume) or path.startswith("/") or ".." in path.split("/"):
            sys.exit(f"evidence {item}: only paths on our miles-/mdev-/rl- volumes")
        dest = work / "evidence" / volume / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        sh(["modal", "volume", "get", "--force", volume, path, str(dest)], timeout=600)
        out.append(str(dest))
    return out


def gh_api(*args, input=None):
    out = sh(["gh", "api", *args], input=input).stdout
    return json.loads(out) if out.strip() else None


def diff_lines(n):
    """{path: lines a review comment may anchor to}: the head-side lines of each file's hunks on GitHub."""
    out = {}
    for page in gh_api("--paginate", "--slurp", f"repos/{REPO_SLUG}/pulls/{n}/files"):
        for f in page:
            lines, right = set(), 0
            for ln in (f.get("patch") or "").splitlines():
                if m := re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", ln):
                    right = int(m.group(1))
                elif not ln.startswith("-"):
                    lines.add(right)
                    right += 1
            out[f["filename"]] = lines
    return out


def line_comment(f):
    body = f"**{f['severity'].capitalize()}** · {f['title']}\n\n{f['evidence'].strip()}\n\n"
    c = {"path": f["location"], "line": f["end_line"] if f["exact"] else f["line"], "side": "RIGHT"}
    if f["exact"] and f["end_line"] > f["line"]:
        c |= {"start_line": f["line"], "start_side": "RIGHT"}
    return c | {"body": body + suggestion_md(f) + "\n" + LINE_MARK}


def post(meta, r, verdict, deep, summary_for, session):
    """Update the summary and post findings inline; summary_for(inline) returns (round, summary)."""
    n, head = meta["number"], meta["headRefOid"]
    me = gh_api("user")["login"]
    ours = [
        c
        for page in gh_api("--paginate", "--slurp", f"repos/{REPO_SLUG}/issues/{n}/comments")
        for c in page
        if c["user"]["login"] == me and ROUND.search(c["body"])
    ]
    ours.sort(key=lambda c: c["created_at"])  # collapsing an older summary bumps its updated_at
    lines = diff_lines(n)

    def on_diff(f):
        span = range(f["line"], (f["end_line"] if f["exact"] else f["line"]) + 1)
        return f["line"] > 0 and all(x in lines.get(f["location"], ()) for x in span)

    inline = [i for i, f in enumerate(r["findings"]) if on_diff(f)]
    rnd, summary = summary_for(inline)
    marker = MARKER.format(
        head=head,
        mode="deep" if deep else "normal",
        verdict=verdict.replace(" ", "-"),
        round=rnd,
        session=session or "none",
        rules=meta["rules"],
    )
    body = f"{marker}\n{summary}"
    if ours:
        summary_c = gh_api(
            "-X", "PATCH", f"repos/{REPO_SLUG}/issues/comments/{ours[-1]['id']}", "-F", "body=@-", input=body
        )
    else:
        summary_c = gh_api("-X", "POST", f"repos/{REPO_SLUG}/issues/{n}/comments", "-F", "body=@-", input=body)
    set_status(meta, "success" if verdict == "ready" else "failure", f"round {rnd}: {verdict}", summary_c["html_url"])
    try:  # tidying earlier rounds is best effort: the round itself is already posted
        resolve_old_threads(n, me)
    except SystemExit as e:
        print(f"could not resolve earlier line threads: {e}", file=sys.stderr)
    note = (
        f"**Codex review, round {rnd}** at `{head[:8]}`: {verdict} ({counts(r)}). [Summary]({summary_c['html_url']})"
    )
    review = {"commit_id": head, "event": "COMMENT", "body": note}
    comments = [line_comment(r["findings"][i]) for i in inline]
    try:
        gh_api(
            "-X",
            "POST",
            f"repos/{REPO_SLUG}/pulls/{n}/reviews",
            "--input",
            "-",
            input=json.dumps(review | {"comments": comments}),
        )
    except SystemExit:  # GitHub refused a line; the summary then carries every finding in full
        rnd, summary = summary_for([])
        gh_api(
            "-X",
            "PATCH",
            f"repos/{REPO_SLUG}/issues/comments/{summary_c['id']}",
            "-F",
            "body=@-",
            input=f"{marker}\n{summary}",
        )
        gh_api("-X", "POST", f"repos/{REPO_SLUG}/pulls/{n}/reviews", "--input", "-", input=json.dumps(review))
    for c in ours[:-1]:
        q = "mutation($id:ID!){minimizeComment(input:{subjectId:$id,classifier:OUTDATED}){clientMutationId}}"
        try:
            graphql(q, id=c["node_id"])
        except SystemExit as e:
            print(f"could not collapse an earlier summary: {e}", file=sys.stderr)


def graphql(query, **variables):
    args = ["graphql", "-f", f"query={query}"] + [f"-F{k}={v}" for k, v in variables.items()]
    return gh_api(*args)


def resolve_old_threads(n, me):
    owner, name = REPO_SLUG.split("/")
    q = """query($o:String!,$r:String!,$n:Int!){repository(owner:$o,name:$r){pullRequest(number:$n){
      reviewThreads(last:100){nodes{id isResolved comments(first:1){nodes{author{login} body}}}}}}}"""
    threads = graphql(q, o=owner, r=name, n=n)["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]
    for t in threads:
        first = t["comments"]["nodes"][0] if t["comments"]["nodes"] else {}
        if (
            not t["isResolved"]
            and (first.get("author") or {}).get("login") == me
            and LINE_MARK in first.get("body", "")
        ):
            graphql("mutation($id:ID!){resolveReviewThread(input:{threadId:$id}){clientMutationId}}", id=t["id"])


def run_agent(args, work, prompt, tag, session=None):
    """One agent turn, in `session` if given; returns (answer, usage, the session it ran in)."""
    out, events = work / f"{args.agent}-{tag}.json", work / f"{args.agent}-{tag}.jsonl"
    cmd = AGENTS[args.agent](args, out, session)
    t0 = time.time()
    try:
        p = sh(cmd, timeout=args.timeout, check=False, input=prompt)
    except subprocess.TimeoutExpired:
        sys.exit(f"{args.agent} did not finish within {args.timeout} s")
    events.write_text(p.stdout)
    usage = {}
    for line in p.stdout.splitlines():
        ev = json.loads(line) if line.startswith("{") else {}
        if ev.get("type") == "thread.started":
            session = ev["thread_id"]
        if ev.get("type") == "turn.completed":
            usage = ev.get("usage", {})
        if ev.get("type") == "turn.failed":
            sys.exit(f"{args.agent} failed: {line[:500]}")
    if not out.exists():
        sys.exit(f"{args.agent} wrote no result; see {events}\n{p.stderr[-1500:]}")
    text = out.read_text()
    answer = json.loads(text[text.index("{") : text.rindex("}") + 1])
    return answer, {"seconds": round(time.time() - t0), **usage}, session


def validate(r, ctx, tree):
    """Mechanical protocol checks; returns the violations as sentences the agent can act on."""
    bad = []
    worst = next((s for s in SEVERITY_ORDER[:2] if any(f["severity"] == s for f in r["findings"])), None)
    if r["verdict"] != VERDICT_FOR[worst]:
        bad.append(f"verdict is '{r['verdict']}' but the findings make it '{VERDICT_FOR[worst]}'")
    for i, f in enumerate(r["findings"], 1):
        loc = f["location"]
        if loc in ctx["files"] or loc in ctx.get("context", ()):
            path = tree / loc
            if f["line"] > 0 and path.exists() and f["line"] > len(path.read_text(errors="replace").splitlines()):
                bad.append(f"finding {i}: {loc}:{f['line']} is past the end of the file")
        elif loc != "PR body" and not re.fullmatch(r"commit [0-9a-f]{7,40}", loc):
            bad.append(f"finding {i}: location '{loc}' is not a changed file, 'PR body' or 'commit <sha>'")
        if not f["evidence"].strip() or not f["suggestion"].strip():
            bad.append(f"finding {i}: evidence and suggestion must both be filled")
        if f["exact"] and (loc not in ctx["files"] or not 0 < f["line"] <= f["end_line"]):
            bad.append(f"finding {i}: an exact suggestion needs a changed file and 0 < line <= end_line")
    rules = {(f["rule"], f["severity"]) for f in r["findings"]}
    if "ledger check: ok" not in ctx["checks"]["head"]["ledger"] and ("ledger", "blocking") not in rules:
        bad.append("ledger check fails on the head but there is no blocking 'ledger' finding")
    if ctx["attribution"] and not any(f["rule"] == "attribution" for f in r["findings"]):
        bad.append(f"attribution lines present ({ctx['attribution'][0]!r}) but no 'attribution' finding")
    for node, p in ctx["proofs"].items():
        name = node.rsplit("::", 1)[-1]
        named = any(
            f["rule"] == "test" and name in f["title"] + f["evidence"] + f["suggestion"] for f in r["findings"]
        )
        if p["head"][0] == "pass" and p["base"][0] == "pass" and not named:
            bad.append(f"{name} passes on the base, so it cannot go red, but no 'test' finding names it")
    return bad


def brief(name, text):
    if name == "precommit":
        failed = re.findall(r"^(.+?)\.{3,}Failed$", text, re.M)
        return f"failed: {', '.join(failed)}" if failed else "all hooks pass"
    return text.strip().splitlines()[-1].strip("= ") if text.strip() else "-"


def regressed(name, head, base):
    """Whether a check's brief result is worse on the head than on the base."""
    if name == "ledger":
        return "ok" not in head and "ok" in base
    if name == "precommit":
        return bool(failed_hooks(head) - failed_hooks(base))
    return failures(head) > failures(base)


def failed_hooks(brief_text):
    return set(brief_text.removeprefix("failed: ").split(", ")) if brief_text.startswith("failed") else set()


def failures(brief_text):
    return sum(int(n) for n in re.findall(r"(\d+) (?:failed|errors?)\b", brief_text))


def counts(r):
    n = {s: sum(f["severity"] == s for f in r["findings"]) for s in SEVERITY_ORDER}
    return ", ".join(f"{v} {k}" for k, v in n.items() if v) or "no findings"


def suggestion_md(f):
    if f["exact"]:
        return f"```suggestion\n{f['suggestion'].rstrip()}\n```"
    return f"Suggested (your call): {f['suggestion'].strip()}"


def render(r, meta, base, run_info, violations, ctx, prior=None, inline=()):
    """Render the round summary, listing inline findings and keeping prior rounds' history."""
    rnd = 1
    history = []
    if prior:
        m = ROUND.search(prior)
        rnd = int(m["round"] or 1) + 1 if m else 2
        history = re.findall(r"^- Round \d+ .*$", prior, re.M)
        history += [f"- {t.group(1)}" for t in [THIS_ROUND.search(prior)] if t]
    head, proofs, checks = meta["headRefOid"], ctx["proofs"], ctx["checks"]
    this = f"Round {rnd} at `{head[:8]}`: {r['verdict']} ({counts(r)})"
    url = f"https://github.com/{REPO_SLUG}/blob/{head}/"
    lines = [
        f"### Codex review · {r['verdict'].capitalize()}",
        f"Round {rnd} at `{head[:8]}`{' · deep' if run_info.get('deep') else ''} · {counts(r)}",
        f"<!-- this-round: {this} -->",
        "",
        r["summary"],
        "",
    ]
    if violations:
        lines += [
            "> **Protocol check failed after the retry; read with care:**",
            *[f"> - {v}" for v in violations],
            "",
        ]
    for sev in SEVERITY_ORDER:
        items = [(i, f) for i, f in enumerate(r["findings"]) if f["severity"] == sev]
        if items:
            lines += [f"**{sev.capitalize()}**", ""]
        for i, f in items:
            where = f["location"] + (f":{f['line']}" if f["line"] else "")
            anchor = f"#L{f['line']}" if f["line"] else ""
            link = f"[`{where}`]({url}{f['location']}{anchor})" if f["location"] in ctx["files"] else f"`{where}`"
            if i in inline:
                lines += [f"- **{f['title']}** at {link} (comment on the line)", ""]
                continue
            lines += [f"- **{f['title']}** at {link} ({f['rule']})", "", f"  {f['evidence'].strip()}", ""]
            lines += ["  " + ln for ln in suggestion_md({**f, "exact": False}).splitlines()] + [""]
    fixed = [p["title"] for p in r.get("prior", ()) if p["status"] == "fixed"]
    if fixed:
        lines += [f"Fixed since round {rnd - 1}: {len(fixed)} ({'; '.join(fixed)})", ""]
    if r["handoffs"]:
        lines += ["**For a human**", ""] + [f"- {h['owner']}: {h['note']}" for h in r["handoffs"]] + [""]
    rows = [(k, brief(k, checks["head"][k]), brief(k, checks["base"][k])) for k in ("ledger", "precommit", "pytest")]
    worse = [row for row in rows if regressed(*row)]
    table = ["| Check | Head | Base |", "| --- | --- | --- |"] + [f"| {k} | {h} | {b} |" for k, h, b in rows]
    if proofs:
        table += [""] + ["| New test | Head | Base |", "| --- | --- | --- |"]
        table += [f"| `{n}` | {p['head'][0]} | {p['base'][0]} |" for n, p in proofs.items()]
    if worse:
        lines += ["**Checks that differ from the base**", ""] + table + [""]
    else:
        lines += ["<details><summary>Checks (none worse than the base)</summary>", "", *table, "", "</details>", ""]
    if inline:  # in full here too, so the next round's --prior has the substance
        full = [f"- **{f['title']}** ({f['location']}:{f['line']}): {f['evidence'].strip()} Suggested: {f['suggestion'].strip()}"
                for i, f in enumerate(r["findings"]) if i in inline]  # fmt: skip
        lines += ["<details><summary>Line comments in full</summary>", "", *full, "", "</details>", ""]
    if history:
        lines += ["<details><summary>History</summary>", "", *history, "", "</details>", ""]
    folded = f"; {meta['folded']}" if meta["folded"] else ""
    lines += [
        f"<sub>{run_info['agent']} `{run_info['model']}` effort {run_info['effort']}, protocol `{run_info['protocol']}`, "
        f"{run_info['seconds']} s, {run_info.get('input_tokens', '?')} in / {run_info.get('output_tokens', '?')} out "
        f"tokens ({run_info.get('cached_input_tokens', 0)} cached), attempt {run_info['attempt']}, "
        f"{run_info.get('session', 'new')} session{', checks reused' if run_info.get('checks_reused') else ''}; "
        f"base `{base[:8]}`{folded}</sub>"
    ]
    return rnd, "\n".join(lines) + "\n"


def main():
    signal.signal(signal.SIGTERM, stop)  # finally blocks then end the sandbox
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pr", type=int)
    ap.add_argument(
        "--out", type=Path, default=Path(os.environ.get("MILES_REVIEW_OUT", "~/.cache/miles-review")).expanduser()
    )
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--agent", choices=sorted(AGENTS), default="codex")
    ap.add_argument("--model", help="the agent's model; each adapter has a default")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--python", default=sys.executable, help="interpreter with pytest for miles_ext/tests")
    ap.add_argument("--prior", type=Path, help="an earlier review of this PR; its session is resumed when it can be")
    ap.add_argument("--fresh", action="store_true", help="a full round in a new session, even if one could resume")
    ap.add_argument("--board", type=Path, help="coordination board whose 'reserved' lines hold fork id reservations")
    ap.add_argument("--head", help="review this commit instead of the PR's current head (it must be in --repo)")
    ap.add_argument("--body", type=Path, help="use this file as the PR body instead of the live one")
    ap.add_argument("--result", type=Path, help="use this JSON as the first attempt instead of calling the agent")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--bundle-only", action="store_true", help="run the checks and build the bundle, skip the agent")
    ap.add_argument("--keep", action="store_true", help="keep the checkout after the review")
    ap.add_argument("--deep", action="store_true", help="also review the change itself (SKILL.md section 5)")
    ap.add_argument("--context", nargs="+", default=[], help="deep: more repo paths or globs to inline at the head")
    ap.add_argument(
        "--evidence",
        nargs="+",
        default=[],
        help="deep: logs or result files behind the claims (local or modal://vol/path)",
    )
    ap.add_argument("--checks-on", choices=("local", "modal"), default="local", help="where the PR's own code runs")
    ap.add_argument("--post", action="store_true", help="post the report as a PR comment")
    args = ap.parse_args()
    args.model = args.model or DEFAULT_MODELS[args.agent]
    args.out = args.out.resolve()
    args.repo = Path(sh(["git", "rev-parse", "--show-toplevel"], cwd=args.repo).stdout.strip())

    stamp = time.strftime("%m%d-%H%M")
    work = args.out / f"pr{args.pr}-{stamp}"
    work.mkdir(parents=True, exist_ok=True)
    args.evidence = fetch_evidence(args.evidence, work)
    meta, head, base, tree = prepare(args, work)
    if args.post:
        set_status(meta, "pending", "reviewing")
    posted = False
    try:  # a crash or a stop leaves an error status, not a pending one
        posted = review(args, work, meta, head, base, tree)
    finally:
        if args.post and not posted:
            set_status(meta, "error", "the review did not finish; /review retries")


def review(args, work, meta, head, base, tree):
    """Checks, the agent (resuming the last round's session when it can) and the report; True once posted."""
    bundle, ctx = build_bundle(args, work, meta, head, base, tree)
    if args.bundle_only:
        print(f"bundle: {bundle}")
        return False
    if args.deep:
        ctx["context"] = context_files(args, tree, ctx["files"])
    prior = args.prior.read_text() if args.prior else None
    last = m.groupdict() if prior and (m := ROUND.search(prior)) else None
    meta["rules"] = rules_hash(tree)
    resume = follow_up(args, tree, meta, base, bundle, last)
    full = None
    protocol = (HERE / "SKILL.md").read_text() + SCHEMA.read_text() + PROMPT + Path(__file__).read_text()
    version = sh([args.agent, "--version"], check=False).stdout.strip().split()
    run_info = {
        "agent": f"{args.agent} {version[-1] if version else '?'}",
        "model": args.model,
        "effort": args.effort,
        "deep": args.deep,
        "protocol": hashlib.sha256(protocol.encode()).hexdigest()[:12],
        "checks_reused": ctx["cached"],
    }
    violations, session, resumed = [], None, False
    for attempt in (1, 2):
        retry = "\n\nA previous attempt broke these protocol rules; fix every one:\n" + "\n".join(
            f"- {v}" for v in violations
        )
        if args.result and attempt == 1:
            result, usage = json.loads(args.result.read_text()), {"seconds": 0}
            violations, resume = validate(result, ctx, tree), None  # a break gets one real, fresh retry
            if not violations:
                break
            continue
        # attempt 2 continues attempt 1's session; attempt 1 continues the last round's when follow_up allows
        text, into = (retry.lstrip(), session) if session else (resume, last["session"]) if resume else (None, None)
        if text is not None:
            try:
                result, usage, session = run_agent(args, work, text, f"a{attempt}", into)
                resumed = resumed or bool(last) and into == last["session"]
            except SystemExit as e:
                print(f"could not continue session {into}: {e}; starting a fresh one", file=sys.stderr)
                text = None
        if text is None:
            full = full or build_prompt(args, tree, head, base, bundle, ctx["files"])
            text = full + (retry if violations else "")
            result, usage, session = run_agent(args, work, text, f"a{attempt}")
        (work / f"prompt-a{attempt}.txt").write_text(text)
        resume = None
        violations = validate(result, ctx, tree)
        if not violations:
            break
        print(f"attempt {attempt} broke the protocol: {'; '.join(violations)}", file=sys.stderr)
    run_info.update(usage, attempt=attempt, session="resumed" if resumed else "new")
    if args.deep:  # a new session read every deep input; a resumed one has now read the current ones too
        seen = {**seen_deep(args, session), **{n: digest(t) for n, t in deep_inputs(args, tree, [])}}
        (args.out / f"pr{args.pr}-deep-seen.json").write_text(json.dumps({"session": session, "seen": seen}))
    report = work / "review.md"
    summary_for = lambda inline: render(result, meta, base, run_info, violations, ctx, prior, inline)  # noqa: E731
    report.write_text(summary_for(())[1])
    (work / "review.json").write_text(json.dumps({"run": run_info, "violations": violations, **result}, indent=2))
    if args.post:  # /merge needs "ready"
        post(meta, result, "protocol failed" if violations else result["verdict"], args.deep, summary_for, session)
    if not args.keep:
        shutil.rmtree(tree)
    counts = {s: sum(f["severity"] == s for f in result["findings"]) for s in SEVERITY_ORDER}
    print(f"PR #{args.pr} {head[:8]}: {result['verdict']}; " + ", ".join(f"{n} {s}" for s, n in counts.items()))
    print(f"protocol: {'ok' if not violations else 'VIOLATIONS: ' + '; '.join(violations)}")
    print(f"report: {report}")
    return args.post


class Stopped(BaseException):
    """SIGTERM: unlike a failed resume (SystemExit), nothing may start another agent."""


def stop(*_):
    raise Stopped


if __name__ == "__main__":
    try:
        main()
    except Stopped:
        sys.exit("stopped")
