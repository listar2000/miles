#!/usr/bin/env python3
"""Move the fork onto a newer upstream release: a rebase of the commit series.

    git fetch upstream --tags
    python fork/tools/sync.py v0.1.3            # new branch fork/v0.1.3 = v0.1.3 + our series
    python fork/tools/sync.py v0.1.3 --dry-run  # only show what would move and what to review

Steps it runs:
  1. branch fork/<new> from the current fork branch and ``git rebase --onto
     <new> <old>`` (rerere on). Only fix and hook commits can conflict: they
     come first in the series, each touches a few marked lines, and its message
     says why it exists and when to drop it. If upstream fixed the bug, drop the
     commit (``git rebase --skip``); otherwise re-apply the same intent to the
     new code and keep the ``[fork <id>]`` markers.
  2. a stack bump commit: UPSTREAM_TAG, BASE_IMAGE (digest looked up on Docker
     Hub), FORK_RELEASE reset to r1, FORK_IMAGE emptied. HARBOR_COMMIT is left
     for you: read the new tag's examples/experimental/harbor/README.md.
  3. ``ledger.py check`` against the new base.
  4. prints the upstream diff of every file miles_ext builds on (REVIEW_PATHS):
     extensions cannot conflict textually, so API drift shows up there or in
     the image check, not in the rebase.
Then build the image, run the launcher's check, a save/stop/resume smoke and
one recipe (fork/README.md, "Sync").
"""

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ledger  # noqa: E402

# upstream files and directories our extensions import or mirror; review their diff on every sync
REVIEW_PATHS = [
    "examples/experimental/harbor/",  # harbor_agent_function (loaded by path), README (Harbor pin)
    "miles/rollout/agentic/",  # session URLs, provider credentials
    "miles/rollout/generate_hub/agentic_tool_call.py",  # --custom-generate-function-path we use
    "miles/rollout/filter_hub/dynamic_sampling_filters.py",
    "miles/utils/arguments.py",  # every flag recipe.py emits
    "miles/utils/chat_template_utils/tito_tokenizer.py",  # parsers in presets.py
    "miles/utils/tracking_utils/wandb_utils.py",  # gpu_telemetry mirrors init_wandb_secondary, imports its helpers
    "train_async.py",
    "train.py",
]


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=check, capture_output=True, text=True)


def docker_hub_digest(repo: str, tag: str) -> str:
    token = json.load(urllib.request.urlopen(
        f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"))["token"]
    req = urllib.request.Request(f"https://registry-1.docker.io/v2/{repo}/manifests/{tag}", method="HEAD", headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json",
    })
    return urllib.request.urlopen(req).headers["Docker-Content-Digest"]


def bump_stack(old: str, new: str) -> str:
    path = ROOT / "miles_ext/stack.py"
    s = path.read_text()
    repo = re.search(r'BASE_IMAGE = "([^:]+):', s).group(1)
    try:
        digest = docker_hub_digest(repo, new)
    except Exception as e:  # upstream publishes the image some time after the tag
        print(f"warning: no {repo}:{new} image on Docker Hub yet ({e}); BASE_IMAGE left unpinned")
        digest = "sha256:UNPUBLISHED"
    s = s.replace(f'UPSTREAM_TAG = "{old}"', f'UPSTREAM_TAG = "{new}"')
    s = re.sub(r'BASE_IMAGE = "[^"]+"', f'BASE_IMAGE = "{repo}:{new}@{digest}"', s)
    s = re.sub(r'FORK_RELEASE = "[^"]+"', f'FORK_RELEASE = "fork-{new}-r1"', s)
    s = re.sub(r'FORK_IMAGE = "[^"]*"', 'FORK_IMAGE = ""', s)
    path.write_text(s)
    return f"{repo}:{new}@{digest}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("new_tag")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    old, new = ledger.base_tag(), args.new_tag
    current = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if git("rev-parse", "-q", "--verify", f"refs/tags/{new}", check=False).returncode:
        raise SystemExit(f"tag {new} not found; git fetch upstream --tags")
    if git("status", "--porcelain").stdout.strip():
        raise SystemExit("working tree not clean")

    series = ledger.commits(old)
    print(f"{current}: {len(series)} fork commits on {old}; target {new} "
          f"({git('rev-list', '--count', f'{old}..{new}').stdout.strip()} upstream commits)")
    for c in series:
        risky = "  <- may conflict" if c.kind in ledger.UPSTREAM_KINDS and any(
            git("diff", "--quiet", old, new, "--", f, check=False).returncode for f in c.files) else ""
        print(f"  {c.sha} {c.kind:5} {c.fork_id:9} {c.subject[:70]}{risky}")
    print("\nupstream changes under REVIEW_PATHS:")
    print(git("diff", "--stat", old, new, "--", *REVIEW_PATHS).stdout or "  none")
    if args.dry_run:
        return

    branch = f"fork/{new}"
    git("config", "rerere.enabled", "true")
    git("checkout", "-b", branch)
    rebase = git("rebase", "--onto", new, old, check=False)
    if rebase.returncode:
        print(rebase.stdout, rebase.stderr)
        raise SystemExit(f"rebase stopped on a conflict on {branch}: resolve it using the commit's message "
                         f"(or `git rebase --skip` if upstream fixed the bug), `git rebase --continue`, "
                         f"then `python fork/tools/sync.py {new} --finish`")
    finish(old, new)


def finish(old: str, new: str):
    base_image = bump_stack(old, new)
    git("commit", "-q", "-am", f"stack: rebase onto {new}\n\nBASE_IMAGE {base_image}\n\nFork-Infra: stack-{new}")
    print(f"stack bumped: BASE_IMAGE {base_image}; set HARBOR_COMMIT from {new}'s "
          "examples/experimental/harbor/README.md, then amend that commit")
    sys.exit(subprocess.run([sys.executable, str(ROOT / "fork/tools/ledger.py"), "check", "--base", new]).returncode)


if __name__ == "__main__":
    if "--finish" in sys.argv:
        sys.argv.remove("--finish")
        tag = sys.argv[1]
        finish(ledger.base_tag(), tag)
    else:
        main()
