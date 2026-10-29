#!/usr/bin/env python3
"""Run the independent review of a listar2000/miles PR with a fresh agent, as SKILL.md "Supervised runs" says."""

import argparse
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
MARKER = "<!-- miles-review head={head} mode={mode} verdict={verdict} -->"
CHECK_IMAGE = "docker.io/starlimobius/miles-mobius:main"
EVIDENCE_VOLUMES = re.compile(r"(miles|mdev|rl)-[\w.-]+")  # ours; never another team's volume


def codex(args, out):
    return [
        "codex",
        "exec",
        "-m",
        args.model,
        "-c",
        f"model_reasoning_effort={args.effort}",
        "-s",
        "read-only",
        "--ephemeral",
        "--skip-git-repo-check",
        "--json",
        "--output-schema",
        str(SCHEMA),
        "-o",
        str(out),
        "-",
    ]


# Each adapter returns the command that runs one agent non-interactively and read-only, with the prompt on stdin
# and its final message written to `out`.
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
    runner = ModalRunner(tree, (head, base), work) if args.checks_on == "modal" else LocalRunner(tree, args.python)
    try:  # a stopped review (SIGTERM) still ends its sandbox
        (bundle / "ledger_list.txt").write_text(
            runner.run([runner.python, "fork/tools/ledger.py", "list"], check=True).stdout
        )
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
        checks = {"head": run_checks(runner, head, files), "base": run_checks(runner, base, files)}
        runner.run(["git", "checkout", "-q", head], check=True)
        for name in ("ledger", "precommit", "pytest"):
            (bundle / f"{name}.txt").write_text(
                f"== head {head[:8]}\n{checks['head'][name]}\n\n== base {base[:8]}\n{checks['base'][name]}\n"
            )
        nodes = new_tests(diff)
        proofs = prove_tests(runner, base, files, nodes)
    finally:
        runner.close()
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
    return bundle, {"files": files, "checks": checks, "attribution": hits, "proofs": proofs}


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
    if args.deep:
        # context files and evidence have no stand-in in the diff, so they are never dropped
        sections += [(f"head:{f}", (tree / f).read_text(errors="replace")) for f in context_files(args, tree, files)]
        sections += [(f"evidence/{name}", text) for name, text in evidence(args)]
    body = "\n".join(f"\n===== {name} =====\n{text}" for name, text in sections + head_files)
    if len(body) > MAX_PROMPT_CHARS:  # too big for the agent: the diff in bundle/ has to stand in for full files
        body = "\n".join(f"\n===== {name} =====\n{text}" for name, text in sections)
    if len(body) > MAX_PROMPT_CHARS:
        sys.exit(
            f"the bundle{', context and evidence' if args.deep else ''} alone are {len(body)} characters, over "
            f"{MAX_PROMPT_CHARS}; narrow them, split the PR or review it by hand"
        )
    return PROMPT.format(pr=args.pr, head=head, base=base) + (DEEP if args.deep else "") + body


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


def post(meta, report, verdict, deep):
    """The review as a PR comment, under the marker the review daemon reads back."""
    marker = MARKER.format(
        head=meta["headRefOid"], mode="deep" if deep else "normal", verdict=verdict.replace(" ", "-")
    )
    body = f"{marker}\n**Independent review (Codex, run by the review daemon)**\n\n{report.read_text()}"
    body_file = report.with_suffix(".comment.md")
    body_file.write_text(body)
    sh(["gh", "pr", "comment", str(meta["number"]), "-R", REPO_SLUG, "--body-file", str(body_file)])


def run_agent(args, work, prompt, tag):
    out, events = work / f"{args.agent}-{tag}.json", work / f"{args.agent}-{tag}.jsonl"
    cmd = AGENTS[args.agent](args, out)
    t0 = time.time()
    try:
        p = sh(cmd, timeout=args.timeout, check=False, input=prompt)
    except subprocess.TimeoutExpired:
        sys.exit(f"{args.agent} did not finish within {args.timeout} s")
    events.write_text(p.stdout)
    usage = {}
    for line in p.stdout.splitlines():
        ev = json.loads(line) if line.startswith("{") else {}
        if ev.get("type") == "turn.completed":
            usage = ev.get("usage", {})
        if ev.get("type") == "turn.failed":
            sys.exit(f"{args.agent} failed: {line[:500]}")
    if not out.exists():
        sys.exit(f"{args.agent} wrote no result; see {events}\n{p.stderr[-1500:]}")
    text = out.read_text()
    return json.loads(text[text.index("{") : text.rindex("}") + 1]), {"seconds": round(time.time() - t0), **usage}


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
        if not f["evidence"].strip() or not f["fix"].strip():
            bad.append(f"finding {i}: evidence and fix must both be filled")
        if re.match(r"\s*(consider|maybe|you could)\b", f["fix"], re.I):
            bad.append(f"finding {i}: fix must be exact text or a diff, not advice")
    rules = {(f["rule"], f["severity"]) for f in r["findings"]}
    if "ledger check: ok" not in ctx["checks"]["head"]["ledger"] and ("ledger", "blocking") not in rules:
        bad.append("ledger check fails on the head but there is no blocking 'ledger' finding")
    if ctx["attribution"] and not any(f["rule"] == "attribution" for f in r["findings"]):
        bad.append(f"attribution lines present ({ctx['attribution'][0]!r}) but no 'attribution' finding")
    for node, p in ctx["proofs"].items():
        name = node.rsplit("::", 1)[-1]
        named = any(f["rule"] == "test" and name in f["title"] + f["evidence"] + f["fix"] for f in r["findings"])
        if p["head"][0] == "pass" and p["base"][0] == "pass" and not named:
            bad.append(f"{name} passes on the base, so it cannot go red, but no 'test' finding names it")
    return bad


def render(r, meta, base, run_info, violations, proofs, checks):
    folded = f" ({meta['folded']})" if meta["folded"] else ""
    lines = [
        f"{r['verdict'].capitalize()}.",
        "",
        f"# PR #{meta['number']} review: {meta['title']}",
        "",
        f"Head `{meta['headRefOid'][:10]}`{folded}, base `{base[:10]}`. "
        f"Reviewer: {run_info['agent']} `{run_info['model']}` "
        f"(effort {run_info['effort']}{', deep review' if run_info.get('deep') else ''}), "
        f"protocol `{run_info['protocol']}`, {run_info['seconds']} s, "
        f"{run_info.get('input_tokens', '?')} in / {run_info.get('output_tokens', '?')} out tokens, "
        f"attempt {run_info['attempt']}.",
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
        items = [f for f in r["findings"] if f["severity"] == sev]
        if not items:
            continue
        lines += [f"## {sev.capitalize()}", ""]
        for f in items:
            where = f["location"] + (f":{f['line']}" if f["line"] else "")
            lines += [
                f"### {f['title']}",
                "",
                f"`{where}` ({f['rule']})",
                "",
                f["evidence"],
                "",
                "Fix:",
                "",
                "```",
                f["fix"].rstrip(),
                "```",
                "",
            ]
    if proofs:
        lines += ["## Test proofs (run by the supervisor)", "", "| Test | Head | Base |", "| --- | --- | --- |"]
        lines += [
            f"| `{n}` | {p['head'][0]} ({p['head'][1]} s) | {p['base'][0]} ({p['base'][1]} s) |"
            for n, p in proofs.items()
        ]
        lines.append("")

    def brief(name, text):
        if name == "precommit":
            failed = re.findall(r"^(.+?)\.{3,}Failed$", text, re.M)
            return f"failed: {', '.join(failed)}" if failed else "all hooks pass"
        return text.strip().splitlines()[-1].strip("= ") if text.strip() else "-"

    lines += ["## Checks (run by the supervisor)", "", "| Check | Head | Base |", "| --- | --- | --- |"]
    lines += [f"| {k} | {brief(k, checks['head'][k])} | {brief(k, checks['base'][k])} |" for k in checks["head"]]
    if r["handoffs"]:
        lines += ["", "## Handoffs", ""] + [f"- **{h['owner']}**: {h['note']}" for h in r["handoffs"]]
    return "\n".join(lines) + "\n"


def main():
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("stopped"))  # finally blocks then end the sandbox
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
    ap.add_argument("--prior", type=Path, help="an earlier review of this PR")
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
    bundle, ctx = build_bundle(args, work, meta, head, base, tree)
    if args.bundle_only:
        print(f"bundle: {bundle}")
        return

    if args.deep:
        ctx["context"] = context_files(args, tree, ctx["files"])
    prompt = build_prompt(args, tree, head, base, bundle, ctx["files"])
    (work / "prompt.txt").write_text(prompt)
    protocol = (HERE / "SKILL.md").read_text() + SCHEMA.read_text() + PROMPT + Path(__file__).read_text()
    version = sh([args.agent, "--version"], check=False).stdout.strip().split()
    run_info = {
        "agent": f"{args.agent} {version[-1] if version else '?'}",
        "model": args.model,
        "effort": args.effort,
        "deep": args.deep,
        "protocol": hashlib.sha256(protocol.encode()).hexdigest()[:12],
    }
    violations = []
    for attempt in (1, 2):
        if violations:
            prompt += "\n\nA previous attempt broke these protocol rules; fix every one:\n" + "\n".join(
                f"- {v}" for v in violations
            )
        if args.result and attempt == 1:
            result, usage = json.loads(args.result.read_text()), {"seconds": 0}
        else:
            result, usage = run_agent(args, work, prompt, f"a{attempt}")
        violations = validate(result, ctx, tree)
        if not violations:
            break
        print(f"attempt {attempt} broke the protocol: {'; '.join(violations)}", file=sys.stderr)
    run_info.update(usage, attempt=attempt)
    report = work / "review.md"
    report.write_text(render(result, meta, base, run_info, violations, ctx["proofs"], ctx["checks"]))
    (work / "review.json").write_text(json.dumps({"run": run_info, "violations": violations, **result}, indent=2))
    if args.post:
        post(meta, report, "protocol failed" if violations else result["verdict"], args.deep)  # /merge needs "ready"
    if not args.keep:
        shutil.rmtree(tree)
    counts = {s: sum(f["severity"] == s for f in result["findings"]) for s in SEVERITY_ORDER}
    print(f"PR #{args.pr} {head[:8]}: {result['verdict']}; " + ", ".join(f"{n} {s}" for s, n in counts.items()))
    print(f"protocol: {'ok' if not violations else 'VIOLATIONS: ' + '; '.join(violations)}")
    print(f"report: {report}")


if __name__ == "__main__":
    main()
