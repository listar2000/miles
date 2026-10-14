#!/usr/bin/env python3
"""Build and push the fork image: upstream's release image plus two layers.

    set -a; . <launch folder>/.env; set +a      # DOCKERHUB_USERNAME / DOCKERHUB_TOKEN
    git push origin fork/v0.1.2                  # the released commit must be on GitHub
    python fork/docker/build.py                  # prints FORK_IMAGE = "docker.io/...@sha256:..."

The image is ``BASE_IMAGE`` (amd64) with every upstream layer untouched, then:

  deps  Harbor (HARBOR_COMMIT) and mini-swe-agent installed into the image's
        Python. Modal runs ``uv pip install`` in a container of the base image;
        the layer is exactly the files of the distributions that install added
        or replaced (their RECORD files), plus whiteouts for the files of the
        versions it removed. Nothing else in the environment can differ.
  src   the fork commit's tree (``git archive``) over /root/miles, the base
        image's own checkout of the upstream tag, plus whiteouts for files the
        fork commit deletes relative to that tag.

then the environment the fork needs (``PYTHONPATH=/root/miles`` so the
top-level ``miles_ext`` imports, release and commit ids) and OCI labels.
``crane`` (go-containerregistry, pinned below and checksummed) writes the
image straight to Docker Hub: the upstream layers are mounted from
radixark/miles, not uploaded, and no Docker daemon is involved.

Why not ``docker build``: no Docker daemon here, CI secrets were not an
option, and Docker inside a Modal Sandbox could not mount the 117-layer base
(overlay mount "invalid argument"). fork/README.md, "The image".
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
CRANE_VERSION = "v0.20.2"
CRANE_SHA256 = "c14340087103ba9dadf61d45acd20675490fd0ccbd56ac7901fc1b502137f44b"  # go-containerregistry_Linux_x86_64.tar.gz
PLATFORM = "linux/amd64"
BUILD_VOLUME = "miles-fork-build"  # layer tarballs in transit; safe to delete

if modal.is_local():
    sys.path.insert(0, str(ROOT))
    from miles_ext import stack

    BASE_IMAGE = stack.BASE_IMAGE
    DEPS_INSTALL = (
        # uv: the Harbor branch has a uv-workspace dependency; --python: the
        # image's interpreter is the /opt/sglang venv, not /usr
        'uv pip install --python "$(command -v python)" '
        f'"harbor[modal] @ git+{stack.HARBOR_REPO}@{stack.HARBOR_COMMIT}" '
        f'"mini-swe-agent=={stack.MINI_SWE_AGENT_VERSION}"'
    )
else:
    BASE_IMAGE = DEPS_INSTALL = "unused-in-container"

app = modal.App("miles-fork-build")
base_image = modal.Image.from_registry(BASE_IMAGE).entrypoint([])
deps_image = base_image.run_commands(DEPS_INSTALL)
volume = modal.Volume.from_name(BUILD_VOLUME, create_if_missing=True)


def _dists() -> dict[str, list[str]]:
    """{dist-info dir: absolute paths in its RECORD} for every installed distribution."""
    import site

    out = {}
    for sp in site.getsitepackages():
        for info in Path(sp).glob("*.dist-info"):
            record = info / "RECORD"
            paths = []
            if record.exists():
                for line in record.read_text().splitlines():
                    rel = line.rsplit(",", 2)[0]
                    if rel:
                        paths.append(os.path.normpath(os.path.join(sp, rel)))
            out[str(info)] = paths
    return out


@app.function(image=base_image, cpu=2, memory=4096, timeout=1200)
def base_dists() -> dict[str, list[str]]:
    return _dists()


@app.function(image=deps_image, cpu=4, memory=8192, timeout=1800, volumes={"/out": volume})
def deps_layer(base: dict[str, list[str]], name: str) -> dict:
    after = _dists()
    added = sorted(set(after) - set(base))
    removed = sorted(set(base) - set(after))
    files = sorted({p for d in added for p in after[d] if os.path.lexists(p)})
    keep = set(files)
    gone = sorted({p for d in removed for p in base[d]} - keep)
    with tarfile.open(f"/out/{name}", "w:gz") as tar:
        dirs = set()
        for path in files:
            for parent in reversed(Path(path).parents[:-1]):
                if str(parent) not in dirs:
                    dirs.add(str(parent))
                    tar.add(str(parent), arcname=str(parent).lstrip("/"), recursive=False)
            tar.add(path, arcname=path.lstrip("/"), recursive=False)
        for path in gone + removed:  # OCI whiteouts: the old version's files and dist-info dirs
            info = tarfile.TarInfo(os.path.join(os.path.dirname(path), ".wh." + os.path.basename(path)).lstrip("/"))
            tar.addfile(info)
    volume.commit()
    return {"added": [Path(d).name for d in added], "removed": [Path(d).name for d in removed],
            "files": len(files), "whiteouts": len(gone) + len(removed)}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True).stdout.strip()


def src_layer(commit: str, path: Path):
    """The fork tree over /root/miles, with whiteouts for files the fork deletes from the upstream tag."""
    archive = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", "--prefix=root/miles/", commit],
                             check=True, capture_output=True).stdout
    deleted = git("diff", "--name-only", "--diff-filter=D", stack.UPSTREAM_TAG, commit).split()
    with tempfile.NamedTemporaryFile(suffix=".tar") as raw:
        raw.write(archive)
        raw.flush()
        with tarfile.open(raw.name) as src, tarfile.open(path, "w:gz") as out:
            for member in src.getmembers():
                if member.type == tarfile.XGLTYPE:  # git's pax header with the commit id
                    continue
                out.addfile(member, src.extractfile(member) if member.isfile() else None)
            for rel in deleted:
                out.addfile(tarfile.TarInfo(f"root/miles/{os.path.dirname(rel)}/.wh.{os.path.basename(rel)}"
                                            .replace("//", "/")))
    return len(deleted)


def crane_binary(workdir: Path) -> Path:
    tgz = workdir / "crane.tgz"
    url = (f"https://github.com/google/go-containerregistry/releases/download/{CRANE_VERSION}/"
           "go-containerregistry_Linux_x86_64.tar.gz")
    tgz.write_bytes(urllib.request.urlopen(url).read())
    if hashlib.sha256(tgz.read_bytes()).hexdigest() != CRANE_SHA256:
        raise SystemExit("crane download checksum mismatch")
    with tarfile.open(tgz) as t:
        t.extract("crane", workdir)
    return workdir / "crane"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", default="HEAD", help="fork commit to release (must be on the fork remote)")
    ap.add_argument("--release", default=stack.FORK_RELEASE, help="image tag, fork-<upstream tag>-r<N>")
    ap.add_argument("--repo", default=None, help="Docker Hub repository (default: $DOCKERHUB_USERNAME/miles)")
    args = ap.parse_args()

    user, token = os.environ.get("DOCKERHUB_USERNAME"), os.environ.get("DOCKERHUB_TOKEN")
    if not user or not token:
        raise SystemExit("DOCKERHUB_USERNAME and DOCKERHUB_TOKEN must be set")
    commit = git("rev-parse", args.commit)
    if not git("branch", "-r", "--contains", commit):
        raise SystemExit(f"{commit[:10]} is not on any remote branch; push it first")
    target = f"docker.io/{args.repo or user + '/miles'}:{args.release}"
    t0 = time.monotonic()

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        crane = crane_binary(work)
        env = {**os.environ, "DOCKER_CONFIG": str(work / "docker")}  # login lives only in this temp dir

        def run_crane(*cargs: str, stdin: str | None = None) -> str:
            return subprocess.run([str(crane), *cargs], env=env, input=stdin, check=True, capture_output=True,
                                  text=True).stdout.strip()

        base_ref = BASE_IMAGE.split("@")[0]
        base_amd64 = f"{base_ref.split(':')[0]}@{run_crane('digest', '--platform', PLATFORM, BASE_IMAGE)}"
        print(f"release {args.release}: commit {commit[:10]} on {base_amd64}", flush=True)

        deps_name = f"deps-{args.release}-{commit[:10]}.tar.gz"
        with modal.enable_output(), app.run():
            summary = deps_layer.remote(base_dists.remote(), deps_name)
        print(f"deps layer: +{summary['added']} -{summary['removed']} "
              f"({summary['files']} files, {summary['whiteouts']} whiteouts)", flush=True)
        deps_tar = work / "deps.tar.gz"
        with deps_tar.open("wb") as f:
            for chunk in volume.read_file(deps_name):
                f.write(chunk)
        volume.remove_file(deps_name)
        src_tar = work / "src.tar.gz"
        n_deleted = src_layer(commit, src_tar)
        print(f"src layer: {commit[:10]} over /root/miles ({n_deleted} whiteouts); "
              f"sizes deps {deps_tar.stat().st_size / 1e6:.1f} MB, src {src_tar.stat().st_size / 1e6:.1f} MB",
              flush=True)

        run_crane("auth", "login", "index.docker.io", "-u", user, "--password-stdin", stdin=token)
        labels = {
            "org.opencontainers.image.source": stack.FORK_REPO,
            "org.opencontainers.image.revision": commit,
            "org.opencontainers.image.version": args.release,
            "org.opencontainers.image.base.name": BASE_IMAGE,
            "org.opencontainers.image.base.digest": base_amd64.split("@")[1],
            "miles.fork.harbor": f"{stack.HARBOR_REPO}@{stack.HARBOR_COMMIT}",
            "miles.fork.mini-swe-agent": stack.MINI_SWE_AGENT_VERSION,
        }
        envs = {"PYTHONPATH": str(stack.MILES_SRC), "MILES_FORK_RELEASE": args.release, "MILES_FORK_COMMIT": commit}
        mutate = ["mutate", base_amd64, "--append", str(deps_tar), "--append", str(src_tar), "-t", target]
        for k, v in envs.items():
            mutate += ["--env", f"{k}={v}"]
        for k, v in labels.items():
            mutate += ["--label", f"{k}={v}"]
        pushed = run_crane(*mutate)
        digest = run_crane("digest", target)
    print(f"pushed {pushed} in {(time.monotonic() - t0) / 60:.1f} min\n"
          f'FORK_IMAGE = "{target}@{digest}"')
    print(json.dumps({"release": args.release, "commit": commit, "image": f"{target}@{digest}"}))


if __name__ == "__main__":
    main()
