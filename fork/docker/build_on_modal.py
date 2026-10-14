#!/usr/bin/env python3
"""Build fork/docker/Dockerfile and push it to Docker Hub, from a Modal Sandbox.

Docker runs inside a Modal Sandbox (``enable_docker``), so the build needs no
local Docker daemon and no CI secrets: Docker Hub credentials go from the local
environment to the sandbox as a Modal Secret and nowhere else. The base image
(~24 GB compressed) is pulled inside Modal; only our new layers are pushed.

    export DOCKERHUB_USERNAME=... DOCKERHUB_TOKEN=...   # e.g. set -a; . ../.env; set +a
    python fork/docker/build_on_modal.py                # builds HEAD, which must be pushed to the fork

Prints the pushed image reference with its digest; put it in
``miles_ext/stack.py:FORK_IMAGE``. Sandbox app: ``miles-fork-build``.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from miles_ext import stack  # noqa: E402

DOCKER_SETUP = (
    # Modal's sandbox kernel has no NAT tables: no bridge network, builds use the host network
    "update-alternatives --set iptables /usr/sbin/iptables-legacy >/dev/null && "
    "update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy >/dev/null && "
    "(dockerd --iptables=false --ip6tables=false --bridge=none >/tmp/dockerd.log 2>&1 &) && "
    "for i in $(seq 60); do docker info >/dev/null 2>&1 && break; sleep 1; done && docker version --format '{{.Server.Version}}'"
)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True).stdout.strip()


def run(sb: modal.Sandbox, script: str) -> str:
    proc = sb.exec("bash", "-c", f"set -euo pipefail; {script}")
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    err = proc.stderr.read()
    if proc.returncode != 0:
        print(err, file=sys.stderr)
        raise SystemExit(f"sandbox step failed (exit {proc.returncode})")
    return "".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default="HEAD", help="fork commit to build (must be on the fork remote)")
    ap.add_argument("--release", default=stack.FORK_RELEASE, help="image tag, fork-<upstream tag>-r<N>")
    ap.add_argument("--repo", default=None, help="Docker Hub repository (default: $DOCKERHUB_USERNAME/miles)")
    args = ap.parse_args()

    user, token = os.environ.get("DOCKERHUB_USERNAME"), os.environ.get("DOCKERHUB_TOKEN")
    if not user or not token:
        raise SystemExit("DOCKERHUB_USERNAME and DOCKERHUB_TOKEN must be set")
    commit = git("rev-parse", args.commit)
    if not git("branch", "-r", "--contains", commit):
        raise SystemExit(f"{commit[:10]} is not on any remote branch; push it first (the build fetches it)")
    image = f"docker.io/{args.repo or user + '/miles'}:{args.release}"
    build_args = {
        "BASE_IMAGE": stack.BASE_IMAGE,
        "HARBOR_REPO": stack.HARBOR_REPO,
        "HARBOR_COMMIT": stack.HARBOR_COMMIT,
        "MINI_SWE_AGENT_VERSION": stack.MINI_SWE_AGENT_VERSION,
        "FORK_REPO": stack.FORK_REPO,
        "FORK_COMMIT": commit,
        "FORK_RELEASE": args.release,
    }
    print(f"building {image} from {commit[:10]} on {stack.BASE_IMAGE.split('@')[0]}", flush=True)

    app = modal.App.lookup("miles-fork-build", create_if_missing=True)
    builder = modal.Image.debian_slim().apt_install("docker.io", "iptables")
    secret = modal.Secret.from_dict({"DOCKERHUB_USERNAME": user, "DOCKERHUB_TOKEN": token})
    sb = modal.Sandbox.create(
        "sleep", "infinity", app=app, image=builder, secrets=[secret],
        cpu=8, memory=32768, timeout=3 * 3600, experimental_options={"enable_docker": True},
    )
    t0 = time.monotonic()
    try:
        run(sb, "mkdir -p /ctx")
        with sb.open("/ctx/Dockerfile", "w") as f:
            f.write((ROOT / "fork/docker/Dockerfile").read_text())
        run(sb, DOCKER_SETUP)
        flags = " ".join(f"--build-arg {k}='{v}'" for k, v in build_args.items())
        run(sb, f"docker build --network host --progress plain {flags} -t {image} /ctx 2>&1")
        run(sb, 'echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin >/dev/null && '
                f"docker push {image} 2>&1 | grep -v -E 'Waiting|Preparing|Layer already exists' || true")
        digest = run(sb, f"docker inspect --format '{{{{index .RepoDigests 0}}}}' {image}").strip()
    finally:
        sb.terminate()
    print(f"\npushed in {(time.monotonic() - t0) / 60:.1f} min\nFORK_IMAGE = \"{image}@{digest.split('@')[1]}\"")


if __name__ == "__main__":
    main()
