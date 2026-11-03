#!/usr/bin/env python3
"""Apply the fork's SGLang patches to their pinned checkout, or fail."""

import subprocess
import sys
from pathlib import Path

# ports first: a fix or hook may edit a file a port adds (fork/README.md, "Rules for ports")
PATCHES = sorted(
    Path(__file__).resolve().parent.glob("*.patch"), key=lambda p: (not p.name.startswith("port-"), p.name)
)


def git(src: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", src, *args], capture_output=True, text=True)


def main() -> int:
    if len(sys.argv) < 3 or not sys.argv[2] or set(sys.argv[3:]) - {"--check"}:  # an empty pin matches any HEAD
        raise SystemExit("usage: apply.py <sglang checkout> <pinned commit> [--check]")
    src, pin, check = sys.argv[1], sys.argv[2], "--check" in sys.argv[3:]
    if not PATCHES:
        print("sglang patches: none")
        return 0
    head = git(src, "rev-parse", "HEAD").stdout.strip()
    if not head or not head.startswith(pin):
        raise SystemExit(
            f"sglang patches: {src} is at {head[:10] or '?'}, the patches are pinned to {pin}; "
            "re-check every patch against this SGLang and update stack.SGLANG_COMMIT"
        )
    # one at a time, in order: a later patch may edit a file an earlier one adds, which a single
    # `git apply` of all files does not see. All or none: a failure reverts the ones applied.
    applied: list[str] = []
    for p in map(str, PATCHES):
        result = git(src, "apply", p)
        if result.returncode:
            for done in reversed(applied):
                git(src, "apply", "-R", done).check_returncode()
            raise SystemExit(f"sglang patches: {Path(p).name} does not apply to SGLang {pin}:\n{result.stderr}")
        applied.append(p)
    if check:
        for done in reversed(applied):
            git(src, "apply", "-R", done).check_returncode()
    print(f"sglang patches: {'checked' if check else 'applied'} {', '.join(p.name for p in PATCHES)} on {head[:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
