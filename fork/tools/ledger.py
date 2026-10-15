#!/usr/bin/env python3
"""The fork's ledger, read from the commit series itself.

Every fork commit carries a ``Fork-<Kind>: <id>`` trailer (Fix, Hook, Infra,
Ext) and, for fixes and hooks, ``Touches:`` (upstream files it edits), ``Evidence:``,
``Upstream-Status:`` and ``Remove-When:``. Git keeps these through a rebase,
so the history is the record and nothing has to be kept in sync by hand.

    python fork/tools/ledger.py list              # table of fork commits since the upstream base
    python fork/tools/ledger.py check             # touch-point audit (exit 1 on any violation)
    python fork/tools/ledger.py markdown          # the table for fork/README.md
    python fork/tools/ledger.py markers           # file-only check (no git history; used in the image)

``check`` enforces the layout rules:
  1. outside the fork's own paths (OWN_PATHS) the only files
     that differ from the upstream base are the ones fixes and hooks declare in ``Touches:``;
  2. every declared file carries a ``[fork <id>]`` marker for each commit touching it;
  3. every fork commit has exactly one ``Fork-*`` trailer; fixes and hooks touch
     only upstream files, infra and extensions only the fork's own paths;
  4. every file in fork/vendored.json (third-party code copied verbatim) still
     has the git blob of its upstream source (also checked by ``markers``);
  5. ids follow their kind (``ID_FORMATS``): ``fix-NNN``, ``hook-NNN``, ``ext-NNN``,
     and a kebab-case name for infra (``infra-NNN``, ``release-<tag>-rN``, or a
     short name such as ``repo-refs``). One id names one change; a change may span
     several commits (an extension grows release by release), and a fix or hook
     id is used by exactly one commit.

The upstream base is ``stack.UPSTREAM_BASE``: the upstream commit ``main`` last
merged (a release tag, or between releases the commit an upstream image was
built from; RFC 0001). Commits reachable from ``stack.UPSTREAM_TAG`` are
upstream's too (a release tag's own commits sit on upstream's release branch),
and merge commits carry no change of ours, so neither is listed.
"""

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWN_PATHS = ("fork/", "miles_ext/", "FORK.md", ".claude/skills/miles-fork/")
KINDS = ("Fix", "Hook", "Infra", "Ext")
UPSTREAM_KINDS = ("Fix", "Hook")  # the only kinds allowed to edit upstream files
MARKER = re.compile(r"\[fork ((?:fix|hook)-\d{3})\]")
ID_FORMATS = {"Fix": r"fix-\d{3}", "Hook": r"hook-\d{3}", "Ext": r"ext-\d{3}", "Infra": r"[a-z0-9]+(?:[.-][a-z0-9]+)*"}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True).stdout


def _stack_value(name: str) -> str:
    tree = ast.parse((ROOT / "miles_ext/stack.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == name:
            return ast.literal_eval(node.value)
    raise SystemExit(f"{name} not found in miles_ext/stack.py")


def base_tag() -> str:
    """The newest upstream release tag merged into the branch."""
    return _stack_value("UPSTREAM_TAG")


def base_ref() -> str:
    """The upstream commit the branch last merged: the tag itself, or a commit after it."""
    return _stack_value("UPSTREAM_BASE")


def own(path: str) -> bool:
    return path.startswith(OWN_PATHS)


@dataclass
class Commit:
    sha: str
    subject: str
    trailers: dict[str, list[str]]
    files: list[str] = field(default_factory=list)

    @property
    def kind(self) -> str | None:
        kinds = [k for k in KINDS if f"Fork-{k}" in self.trailers]
        return kinds[0] if len(kinds) == 1 else None

    @property
    def fork_id(self) -> str:
        return self.trailers.get(f"Fork-{self.kind}", ["?"])[0] if self.kind else "?"

    def get(self, key: str) -> str:
        return ", ".join(self.trailers.get(key, []))


def commits(base: str) -> list[Commit]:
    out = []
    sep = "\x1e"
    log = git("log", "--reverse", "--no-merges", f"--format={sep}%H%x00%s%x00%(trailers:only,unfold)", f"{base}..HEAD",
              f"^{base_tag()}")
    for chunk in log.split(sep)[1:]:
        sha, subject, trailer_text = chunk.split("\x00")
        trailers: dict[str, list[str]] = {}
        for line in trailer_text.strip().splitlines():
            key, _, value = line.partition(":")
            trailers.setdefault(key.strip(), []).append(value.strip())
        files = git("diff-tree", "--no-commit-id", "--name-only", "-r", sha).split()
        out.append(Commit(sha[:10], subject, trailers, files))
    return out


def cmd_list(base: str):
    for c in commits(base):
        print(f"{c.sha}  {c.kind or '???':5} {c.fork_id:10} {c.subject}")


def cmd_markdown(base: str):
    print("| Id | Kind | Change | Upstream files | Upstream status | Remove when |")
    print("| --- | --- | --- | --- | --- | --- |")
    for c in commits(base):
        if c.kind in UPSTREAM_KINDS:
            print(f"| {c.fork_id} | {c.kind.lower()} | {c.subject} | {c.get('Touches')} | {c.get('Upstream-Status')} "
                  f"| {c.get('Remove-When')} |")
        elif c.kind:
            print(f"| {c.fork_id} | {c.kind.lower()} | {c.subject} | none | | |")


def markers_in(path: Path) -> set[str]:
    try:
        return set(MARKER.findall(path.read_text()))
    except (OSError, UnicodeDecodeError):
        return set()


def vendored_problems() -> list[str]:
    """Vendored files must stay byte-identical to their source: a git blob hash, no git needed."""
    manifest = ROOT / "fork/vendored.json"
    problems = []
    for entry in json.loads(manifest.read_text()) if manifest.exists() else []:
        path = ROOT / entry["path"]
        try:
            data = path.read_bytes()
        except OSError:
            problems.append(f"{entry['path']}: listed in fork/vendored.json but missing")
            continue
        blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        if blob != entry["blob"]:
            problems.append(f"{entry['path']}: blob {blob[:10]} is not {entry['upstream_path']}@{entry['commit'][:10]} "
                            f"({entry['blob'][:10]}); vendored files are never edited, adapt in a subclass")
    return problems


def cmd_check(base: str) -> int:
    problems = vendored_problems()
    series = commits(base)
    declared: dict[str, set[str]] = {}
    for c in series:
        if c.kind is None:
            problems.append(f"{c.sha} {c.subject!r}: needs exactly one Fork-Fix/Hook/Infra/Ext trailer")
            continue
        if not re.fullmatch(ID_FORMATS[c.kind], c.fork_id):
            problems.append(f"{c.sha} {c.fork_id!r}: a Fork-{c.kind} id must match {ID_FORMATS[c.kind]}")
        if c.kind in UPSTREAM_KINDS and sum(o.fork_id == c.fork_id for o in series) > 1:
            problems.append(f"{c.sha} {c.fork_id}: a {c.kind.lower()} id is used by one commit only")
        if c.kind in UPSTREAM_KINDS:
            touches = {t.strip() for v in c.trailers.get("Touches", []) for t in v.split(",") if t.strip()}
            for key in ("Evidence", "Upstream-Status", "Remove-When"):
                if key not in c.trailers:
                    problems.append(f"{c.sha} {c.fork_id}: missing {key} trailer")
            for f in c.files:
                if own(f):
                    problems.append(f"{c.sha} {c.fork_id}: a {c.kind.lower()} must not touch fork paths ({f})")
                elif f not in touches:
                    problems.append(f"{c.sha} {c.fork_id}: edits {f} but Touches: does not list it")
            for f in touches:
                declared.setdefault(f, set()).add(c.fork_id)
        else:
            for f in c.files:
                if not own(f):
                    problems.append(f"{c.sha} {c.fork_id}: {c.kind.lower()} commits may only touch "
                                    f"{', '.join(OWN_PATHS)} ({f})")
    changed = [f for f in git("diff", "--name-only", base, "HEAD").split() if not own(f)]
    for f in changed:
        if f not in declared:
            problems.append(f"{f}: differs from {base} but no fix or hook declares it")
    for f, ids in sorted(declared.items()):
        if f not in changed:
            problems.append(f"{f}: declared by {sorted(ids)} but identical to {base} (fix dropped? remove the commit)")
        missing = ids - markers_in(ROOT / f)
        if missing:
            problems.append(f"{f}: no [fork <id>] marker for {sorted(missing)}")
    counts = ", ".join(f"{sum(c.kind == k for c in series)} {k.lower()}" for k in KINDS)
    print(f"base {base}: {len(series)} fork commits ({counts}), "
          f"{len(declared)} upstream files touched: {', '.join(sorted(declared)) or 'none'}")
    for p in problems:
        print("  FAIL", p)
    print("ledger check:", "ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def cmd_markers() -> int:
    found: dict[str, set[str]] = {}
    for path in (ROOT / "miles").rglob("*.py"):
        for fid in markers_in(path):
            found.setdefault(fid, set()).add(str(path.relative_to(ROOT)))
    for fid in sorted(found):
        print(f"{fid}: {', '.join(sorted(found[fid]))}")
    if not found:
        print("no fork markers found: is this the fork's source tree?")
        return 1
    problems = vendored_problems()
    for p in problems:
        print("  FAIL", p)
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "check", "markdown", "markers"])
    ap.add_argument("--base", help="upstream commit or tag the branch last merged (default: UPSTREAM_BASE in "
                    "miles_ext/stack.py)")
    args = ap.parse_args()
    if args.command == "markers":
        sys.exit(cmd_markers())
    base = args.base or base_ref()
    if args.command == "list":
        cmd_list(base)
    elif args.command == "markdown":
        cmd_markdown(base)
    else:
        sys.exit(cmd_check(base))


if __name__ == "__main__":
    main()
