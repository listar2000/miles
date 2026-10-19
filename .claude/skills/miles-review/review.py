#!/usr/bin/env python3
"""Run the independent review of a listar2000/miles PR with a fresh agent, as SKILL.md "Supervised runs" says."""

import argparse
import hashlib
import json
import os
import re
import shutil
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
CONTEXT_FILES = (
    ".claude/skills/miles-review/SKILL.md",
    "fork/README.md",
    "fork/rfcs/0001-main-and-stable-versions.md",
    ".claude/skills/miles-fork/SKILL.md",
    ".github/pull_request_template.md",
)
MAX_PROMPT_CHARS = 900_000  # Codex rejects prompts over 1,048,576 characters
TEST_FILE = re.compile(r"(^|/)(tests?/|test_[^/]*\.py$)")


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


def sh(cmd, cwd=None, timeout=None, check=True, input=None):
    p = subprocess.run(
        cmd,
        cwd=cwd,
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
    return meta, head, base, tree


def run_checks(tree, rev, files, python):
    sh(["git", "checkout", "-q", rev], cwd=tree)
    out = {}
    out["ledger"] = tail(sh([python, "fork/tools/ledger.py", "check"], cwd=tree, check=False), 5)
    present = [f for f in files if (tree / f).exists()]
    if present:
        p = sh(["uvx", "pre-commit", "run", "--files", *present], cwd=tree, timeout=1800, check=False)
        kept = [line for line in tail(p, 400).splitlines() if not line.endswith(("Passed", "Skipped"))]
        out["precommit"] = "\n".join(kept + [f"exit {p.returncode}" + ("" if p.returncode else ": all hooks passed")])
        sh(["git", "checkout", "-q", "--", "."], cwd=tree)
    else:
        out["precommit"] = "(no changed files at this revision)"
    p = sh(
        [python, "-m", "pytest", "miles_ext/tests", "-q", "-p", "no:cacheprovider", "--continue-on-collection-errors"],
        cwd=tree,
        timeout=1800,
        check=False,
    )
    out["pytest"] = tail(p, 15)
    sh(["git", "clean", "-fdq"], cwd=tree)
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


def pytest_node(tree, node, python):
    t0 = time.time()
    p = sh([python, "-m", "pytest", node, "-q", "-p", "no:cacheprovider"], cwd=tree, timeout=600, check=False)
    result = {0: "pass", 1: "fail"}.get(p.returncode, "error")
    return result, round(time.time() - t0, 1), tail(p, 3)


def prove_tests(tree, base, files, nodes, python):
    """Run each new test on the head, then with the base's version of every changed production file."""
    proofs = {node: {"head": pytest_node(tree, node, python)} for node in nodes}
    for f in files:
        if TEST_FILE.search(f):
            continue
        if sh(["git", "cat-file", "-e", f"{base}:{f}"], cwd=tree, check=False).returncode == 0:
            sh(["git", "checkout", "-q", base, "--", f], cwd=tree)
        elif (tree / f).exists():
            (tree / f).unlink()
    for node in nodes:
        proofs[node]["base"] = pytest_node(tree, node, python)
    sh(["git", "reset", "-q", "--hard"], cwd=tree)
    sh(["git", "clean", "-fdq"], cwd=tree)
    return proofs


def build_bundle(args, work, meta, head, base, tree):
    bundle = work / "bundle"
    bundle.mkdir(exist_ok=True)
    git = lambda *a: sh(["git", *a], cwd=tree).stdout  # noqa: E731
    diff = git("diff", base, head)
    files = git("diff", "--name-only", base, head).split()
    commits = git("log", "--format=%h %s%n%b%n%(trailers:only)---", f"{base}..{head}")
    (bundle / "pr.json").write_text(json.dumps(meta, indent=2))
    (bundle / "commits.txt").write_text(commits)
    (bundle / "diff.patch").write_text(diff)
    (bundle / "diffstat.txt").write_text(git("diff", "--stat", base, head))
    (bundle / "files.txt").write_text("\n".join(files) + "\n")
    (bundle / "ledger_list.txt").write_text(sh([args.python, "fork/tools/ledger.py", "list"], cwd=tree).stdout)
    if args.board:
        reserved = [line for line in args.board.read_text().splitlines() if re.search(r"reserv", line, re.I)]
        (bundle / "reserved_ids.txt").write_text("\n".join(reserved) or "none")
    checks = {"head": run_checks(tree, head, files, args.python), "base": run_checks(tree, base, files, args.python)}
    sh(["git", "checkout", "-q", head], cwd=tree)
    for name in ("ledger", "precommit", "pytest"):
        (bundle / f"{name}.txt").write_text(
            f"== head {head[:8]}\n{checks['head'][name]}\n\n== base {base[:8]}\n{checks['base'][name]}\n"
        )
    nodes = new_tests(diff)
    proofs = prove_tests(tree, base, files, nodes, args.python)
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
    body = "\n".join(f"\n===== {name} =====\n{text}" for name, text in sections + head_files)
    if len(body) > MAX_PROMPT_CHARS:  # too big for the agent: the diff in bundle/ has to stand in for full files
        body = "\n".join(f"\n===== {name} =====\n{text}" for name, text in sections)
    if len(body) > MAX_PROMPT_CHARS:
        sys.exit(
            f"the bundle alone is {len(body)} characters, over {MAX_PROMPT_CHARS}; split the PR or review it by hand"
        )
    return PROMPT.format(pr=args.pr, head=head, base=base) + body


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
        if loc in ctx["files"]:
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


def render(r, meta, head, base, run_info, violations, proofs, checks):
    lines = [
        f"{r['verdict'].capitalize()}.",
        "",
        f"# PR #{meta['number']} review: {meta['title']}",
        "",
        f"Head `{head[:10]}`, base `{base[:10]}`. Reviewer: {run_info['agent']} `{run_info['model']}` "
        f"(effort {run_info['effort']}), protocol `{run_info['protocol']}`, {run_info['seconds']} s, "
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
    args = ap.parse_args()
    args.model = args.model or DEFAULT_MODELS[args.agent]
    args.out = args.out.resolve()
    args.repo = Path(sh(["git", "rev-parse", "--show-toplevel"], cwd=args.repo).stdout.strip())

    stamp = time.strftime("%m%d-%H%M")
    work = args.out / f"pr{args.pr}-{stamp}"
    work.mkdir(parents=True, exist_ok=True)
    meta, head, base, tree = prepare(args, work)
    bundle, ctx = build_bundle(args, work, meta, head, base, tree)
    if args.bundle_only:
        print(f"bundle: {bundle}")
        return

    prompt = build_prompt(args, tree, head, base, bundle, ctx["files"])
    (work / "prompt.txt").write_text(prompt)
    protocol = (HERE / "SKILL.md").read_text() + SCHEMA.read_text() + PROMPT + Path(__file__).read_text()
    version = sh([args.agent, "--version"], check=False).stdout.strip().split()
    run_info = {
        "agent": f"{args.agent} {version[-1] if version else '?'}",
        "model": args.model,
        "effort": args.effort,
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
    report.write_text(render(result, meta, head, base, run_info, violations, ctx["proofs"], ctx["checks"]))
    (work / "review.json").write_text(json.dumps({"run": run_info, "violations": violations, **result}, indent=2))
    if not args.keep:
        shutil.rmtree(tree)
    counts = {s: sum(f["severity"] == s for f in result["findings"]) for s in SEVERITY_ORDER}
    print(f"PR #{args.pr} {head[:8]}: {result['verdict']}; " + ", ".join(f"{n} {s}" for s, n in counts.items()))
    print(f"protocol: {'ok' if not violations else 'VIOLATIONS: ' + '; '.join(violations)}")
    print(f"report: {report}")


if __name__ == "__main__":
    main()
