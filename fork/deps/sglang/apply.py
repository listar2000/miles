#!/usr/bin/env python3
"""Apply the fork's SGLang patches to their pinned checkout, or fail."""

import subprocess
import sys
from pathlib import Path

PATCHES = sorted(Path(__file__).resolve().parent.glob("*.patch"))


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
    files = [str(p) for p in PATCHES]
    result = git(src, "apply", "--check", *files)
    if result.returncode:
        raise SystemExit(f"sglang patches: they do not apply to SGLang {pin}:\n{result.stderr}")
    if not check:
        git(src, "apply", *files).check_returncode()
    print(f"sglang patches: {'checked' if check else 'applied'} {', '.join(p.name for p in PATCHES)} on {head[:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
