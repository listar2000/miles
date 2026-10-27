#!/usr/bin/env python3
"""The fork's ledger, read from the commit series itself.

Every fork commit carries a ``Fork-<Kind>: <id>`` trailer (Fix, Hook, Infra,
Ext) and, for fixes and hooks, ``Touches:`` (upstream files it edits), ``Evidence:``,
``Upstream-Status:`` and ``Remove-When:``. Git keeps these through a rebase,
so the history is the record and nothing has to be kept in sync by hand.

    python fork/tools/ledger.py list              # table of fork commits since the upstream base
    python fork/tools/ledger.py check             # touch-point audit (exit 1 on any violation)
    python fork/tools/ledger.py markdown          # the table for fork/README.md
    python fork/tools/ledger.py markers           # file-only check (no git history; used in the image,
                                                  # where it also finds each SGLang patch's markers in SGLang)

``check`` enforces the layout rules:
  1. outside the fork's own paths (OWN_PATHS) the only files
     that differ from the upstream base are the ones fixes and hooks declare in ``Touches:``;
  2. every declared file carries a ``[fork <id>]`` marker for each commit touching it;
  3. every fork commit has exactly one ``Fork-*`` trailer; fixes and hooks touch
     only upstream files, infra and extensions only the fork's own paths;
  4. every file in fork/vendored.json (third-party code copied verbatim) still
     has the git blob of its upstream source (also checked by ``markers``);
  5. an SGLang fix or hook is one patch, ``fork/deps/sglang/<id>-<slug>.patch``,
     added or changed only by its own commit, which lists every file the patch
     edits as ``Touches: sglang:<path>``; each of those files gets a line with its
     ``[fork <id>]`` marker. Otherwise only a ``sync-`` commit changes or deletes a patch;
  6. ids follow their kind (``ID_FORMATS``): ``fix-NNN``, ``hook-NNN``, ``ext-NNN``,
     and a kebab-case name for infra (``infra-NNN``, ``release-<tag>-rN``, or a
     short name such as ``repo-refs``). One id names one change; a change may span
     several commits (an extension grows release by release), and a fix or hook
     id is used by exactly one commit;
  7. a sync that drops a fix or hook because upstream fixed it says so with a
     ``Retires: <id>`` trailer on one of its ``sync-`` commits. A fix or hook commit
     that removes an earlier fix or hook of ours retires it the same way. A fix or hook
     commit that removes an earlier change's code from some files only says so with
     ``Releases: <id> <file>, ...``. A retired change's files and markers are no longer
     required (its commit stays in the history, listed as retired), nor a released
     id's marker in that file, nor a retiring or releasing commit's own marker in a
     file where it only deleted lines.

The upstream base is ``stack.UPSTREAM_BASE``: the upstream commit ``main`` last
merged (a release tag, or between releases the commit an upstream image was
built from; RFC 0001). Commits reachable from ``stack.UPSTREAM_TAG`` or an earlier
merged release tag are upstream's too (a release tag's own commits sit on upstream's
release branch), and merge commits carry no change of ours, so neither is listed.
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
OWN_PATHS = (
    "fork/",
    "miles_ext/",
    "FORK.md",
    ".claude/skills/miles-fork/",
    ".claude/skills/miles-review/",
    ".claude/skills/miles-use/",
    ".claude/rules/fork.md",
    ".github/ISSUE_TEMPLATE/",
    ".github/pull_request_template.md",
    ".github/workflows/miles-mobius-image.yml",
)
KINDS = ("Fix", "Hook", "Infra", "Ext")
UPSTREAM_KINDS = ("Fix", "Hook")  # the only kinds allowed to edit upstream files
MARKER = re.compile(r"\[fork ((?:fix|hook)-\d{3})\]")
SGLANG_PATCHES = "fork/deps/sglang/"
SGLANG = "sglang:"  # Touches: prefix for a file of the SGLang checkout (stack.SGLANG_SRC)
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


def require_full_history():
    """Exit with the command to run when the clone is shallow or lacks the upstream tag."""
    if git("rev-parse", "--is-shallow-repository").strip() == "true":
        raise SystemExit("ledger: shallow clone; run `git fetch --unshallow --tags origin` first")
    if not git("tag", "--list", base_tag()).strip():
        raise SystemExit(f"ledger: tag {base_tag()} missing; run `git fetch --tags origin` first")


def own(path: str) -> bool:
    return path.startswith(OWN_PATHS)


def sglang_patch_id(path: str) -> str | None:
    """The fork id of an SGLang patch file (``fork/deps/sglang/fix-001-x.patch`` -> ``fix-001``)."""
    m = re.fullmatch(re.escape(SGLANG_PATCHES) + r"((?:fix|hook)-\d{3})-[a-z0-9-]+\.patch", path)
    return m.group(1) if m else None


def patch_files(text: str, fork_id: str) -> dict[str, bool]:
    """{file a unified diff edits: whether its added lines carry the ``[fork <fork_id>]`` marker}."""
    out: dict[str, bool] = {}
    current = None
    for line in text.splitlines():
        if line.startswith("diff --git "):
            current = line.split(" b/", 1)[1]
            out[current] = False
        elif current and line.startswith("+") and not line.startswith("+++") and fork_id in MARKER.findall(line):
            out[current] = True
    return out


@dataclass
class Commit:
    sha: str
    subject: str
    trailers: dict[str, list[str]]
    files: list[str] = field(default_factory=list)
    deleted_only: set[str] = field(default_factory=set)  # files this commit only deleted lines from

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
    # every upstream release tag merged so far: a release's own commits sit on its release branch,
    # which the next release does not contain
    tags = {base_tag(), *git("tag", "--merged", "HEAD", "--list", "v[0-9]*").split()}
    log = git("log", "--reverse", "--no-merges", f"--format={sep}%H%x00%s%x00%(trailers:only,unfold)", f"{base}..HEAD",
              *(f"^{t}" for t in sorted(tags)))
    for chunk in log.split(sep)[1:]:
        sha, subject, trailer_text = chunk.split("\x00")
        trailers: dict[str, list[str]] = {}
        for line in trailer_text.strip().splitlines():
            key, _, value = line.partition(":")
            trailers.setdefault(key.strip(), []).append(value.strip())
        files = git("diff-tree", "--no-commit-id", "--name-only", "-r", sha).split()
        deleted_only = set()
        for row in git("diff-tree", "--no-commit-id", "--numstat", "-r", sha).splitlines():
            added, _, path = row.split("\t", 2)
            if added == "0":
                deleted_only.add(path)
        out.append(Commit(sha[:10], subject, trailers, files, deleted_only))
    return out


def released(series: list[Commit]) -> set[tuple[str, str]]:
    """{(fix or hook id, file)}: files whose code of that id a later commit removed, from ``Releases:`` trailers."""
    out = set()
    for c in series:
        for v in c.trailers.get("Releases", []):
            for item in (t.strip() for t in v.split(",")):
                if item:
                    fid, _, path = item.partition(" ")
                    out.add((fid, path.strip()))
    return out


def retired(series: list[Commit]) -> dict[str, str]:
    """{fix or hook id: the commit id that retired it}, from ``Retires:`` trailers."""
    out = {}
    for c in series:
        for v in c.trailers.get("Retires", []):
            for fid in (t.strip() for t in v.split(",")):
                if fid:
                    out[fid] = c.fork_id
    return out


def cmd_list(base: str):
    series = commits(base)
    gone = retired(series)
    for c in series:
        note = f"  (retired by {gone[c.fork_id]})" if c.fork_id in gone else ""
        print(f"{c.sha}  {c.kind or '???':5} {c.fork_id:10} {c.subject}{note}")


def cmd_markdown(base: str):
    print("| Id | Kind | Change | Upstream files | Upstream status | Remove when |")
    print("| --- | --- | --- | --- | --- | --- |")
    series = commits(base)
    gone = retired(series)
    for c in series:
        if c.kind in UPSTREAM_KINDS:
            status = f"retired by {gone[c.fork_id]}" if c.fork_id in gone else c.get("Upstream-Status")
            print(f"| {c.fork_id} | {c.kind.lower()} | {c.subject} | {c.get('Touches')} | {status} "
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
    sglang: dict[str, set[str]] = {}  # fork id -> the SGLang files its commit declares
    gone = retired(series)
    upstream_ids = {c.fork_id for c in series if c.kind in UPSTREAM_KINDS}
    for c in series:
        if c.kind is None:
            problems.append(f"{c.sha} {c.subject!r}: needs exactly one Fork-Fix/Hook/Infra/Ext trailer")
            continue
        if "Retires" in c.trailers and not (c.kind == "Infra" and c.fork_id.startswith("sync-")) and (
            c.kind not in UPSTREAM_KINDS
        ):
            problems.append(f"{c.sha} {c.fork_id}: only a sync-, fix or hook commit may carry a Retires: trailer")
        if "Releases" in c.trailers and c.kind not in UPSTREAM_KINDS:
            problems.append(f"{c.sha} {c.fork_id}: only a fix or hook commit may carry a Releases: trailer")
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
                if sglang_patch_id(f) == c.fork_id:
                    continue
                if own(f):
                    problems.append(f"{c.sha} {c.fork_id}: a {c.kind.lower()} must not touch fork paths ({f})")
                elif f not in touches:
                    problems.append(f"{c.sha} {c.fork_id}: edits {f} but Touches: does not list it")
            sglang_touches = {t.removeprefix(SGLANG) for t in touches if t.startswith(SGLANG)}
            if sglang_touches:
                sglang[c.fork_id] = sglang_touches
                if len(c.files) != 1 or sglang_patch_id(c.files[0]) != c.fork_id:
                    problems.append(
                        f"{c.sha} {c.fork_id}: an SGLang fix or hook commit touches exactly one file, "
                        f"{SGLANG_PATCHES}{c.fork_id}-<slug>.patch"
                    )
            if c.fork_id in gone:
                continue
            removes = "Retires" in c.trailers or "Releases" in c.trailers
            for f in touches - {SGLANG + t for t in sglang_touches}:
                if removes and f in c.deleted_only:
                    continue  # a pure deletion leaves nothing to mark
                declared.setdefault(f, set()).add(c.fork_id)
        else:
            for f in c.files:
                if not own(f):
                    problems.append(f"{c.sha} {c.fork_id}: {c.kind.lower()} commits may only touch "
                                    f"{', '.join(OWN_PATHS)} ({f})")
                elif f.startswith(SGLANG_PATCHES) and f.endswith(".patch") and not c.fork_id.startswith("sync-"):
                    problems.append(
                        f"{c.sha} {c.fork_id}: only its own fix or hook commit or a sync- commit "
                        f"changes an SGLang patch ({f})"
                    )
    for fid, f in sorted(released(series)):
        if fid not in upstream_ids:
            problems.append(f"Releases: {fid} {f} names no fix or hook since {base}")
        elif fid not in declared.get(f, set()) and fid not in gone:
            problems.append(f"Releases: {fid} {f}, but {fid} does not touch {f}")
        declared.get(f, set()).discard(fid)
    declared = {f: ids for f, ids in declared.items() if ids}
    for fid in sorted(set(gone) - upstream_ids):
        problems.append(f"Retires: {fid} (in {gone[fid]}) names no fix or hook since {base}")
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
    patches: dict[str, Path] = {}
    for path in sorted((ROOT / SGLANG_PATCHES).glob("*.patch")):
        rel, fid = path.relative_to(ROOT), sglang_patch_id(str(path.relative_to(ROOT)))
        if fid in patches:
            problems.append(f"{rel}: a second patch for {fid} ({patches[fid].name})")
            continue
        if fid in gone:
            problems.append(f"{rel}: {fid} is retired by {gone[fid]}; delete the patch in that sync")
            continue
        if fid not in sglang:
            problems.append(
                f"{rel}: not named <fix|hook>-NNN-<slug>.patch, or no commit of that id declares "
                "its files (Touches: sglang:<path>)"
            )
            continue
        patches[fid] = path
        edited = patch_files(path.read_text(), fid)
        if set(edited) != sglang[fid]:
            problems.append(f"{rel}: edits {sorted(edited)} but {fid} declares {sorted(sglang[fid])}")
        unmarked = sorted(f for f, marked in edited.items() if not marked)
        if unmarked:
            problems.append(f"{rel}: no added [fork {fid}] line in {unmarked}")
    counts = ", ".join(f"{sum(c.kind == k for c in series)} {k.lower()}" for k in KINDS)
    print(
        f"base {base}: {len(series)} fork commits ({counts}{f', {len(set(gone) & upstream_ids)} retired' if gone else ''}), "
        f"{len(declared)} upstream files touched: {', '.join(sorted(declared)) or 'none'}; "
        f"SGLang patches: {', '.join(sorted(patches)) or 'none'}"
    )
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
    problems = vendored_problems() + sglang_marker_problems()
    for p in problems:
        print("  FAIL", p)
    return 1 if problems else 0


def sglang_marker_problems() -> list[str]:
    """In an image: every SGLang patch's marker is in each SGLang file it edits."""
    src = Path(_stack_value("SGLANG_SRC"))
    patches = sorted((ROOT / SGLANG_PATCHES).glob("*.patch"))
    if patches and not src.is_dir():
        return [f"{src}: missing, so {len(patches)} SGLang patch(es) cannot be checked; run this in the image"]
    problems = []
    for path in patches:
        fid = sglang_patch_id(str(path.relative_to(ROOT)))
        for f in patch_files(path.read_text(), fid):
            if fid in markers_in(src / f):
                print(f"{fid}: {src / f}")
            else:
                problems.append(f"{src / f}: no [fork {fid}] marker; {path.name} is not applied in this image")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "check", "markdown", "markers"])
    ap.add_argument("--base", help="upstream commit or tag the branch last merged (default: UPSTREAM_BASE in "
                    "miles_ext/stack.py)")
    args = ap.parse_args()
    if args.command == "markers":
        sys.exit(cmd_markers())
    require_full_history()
    base = args.base or base_ref()
    if args.command == "list":
        cmd_list(base)
    elif args.command == "markdown":
        cmd_markdown(base)
    else:
        sys.exit(cmd_check(base))


if __name__ == "__main__":
    main()
