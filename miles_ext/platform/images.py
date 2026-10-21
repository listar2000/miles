"""Images the platform functions run in."""

import os

import modal

from miles_ext import stack

IMAGE_ENV = "MILES_EXT_IMAGE"  # the registry image a container runs (unset when Modal built it)
TRAINER_ENV = "MILES_MOBIUS_TRAINER"  # what the run asked for: latest or stable-YYYY-MM-DD (set by launch)
# set by fork/docker/build.py in the image; images built before 2026-10-15 carry only the MILES_FORK_* pair
VERSION_ENVS = ("MILES_MOBIUS_VERSION", "MILES_FORK_RELEASE")
COMMIT_ENVS = ("MILES_MOBIUS_COMMIT", "MILES_FORK_COMMIT")


def _first_env(names) -> str | None:
    return next((os.environ[n] for n in names if os.environ.get(n)), None)


def image_version() -> str | None:
    """The miles-mobius version this container runs, e.g. ``main@1a2b3c4d5e`` (or an old ``fork-v0.1.2-rN``)."""
    return _first_env(VERSION_ENVS)


def image_commit() -> str:
    """The miles-mobius commit this container runs (/root/miles/.git still describes upstream's checkout)."""
    return _first_env(COMMIT_ENVS) or "unknown"


_COMMIT_PIN = "from miles_ext.stack import SGLANG_COMMIT as c; print(c)"  # the checked-out commit's pin


def fork_image(commit: str | None = None, image: str | None = None) -> modal.Image:
    """The published fork image: ``image`` (a ``repo@sha256:...`` from ``versions.resolve``)
    or else ``stack.FORK_IMAGE``; or the same image built by Modal from a pushed fork
    commit (what ``fork/docker/build.py`` would publish)."""
    if commit and image:
        raise ValueError("pass a fork commit or an image, not both")
    if image:
        if "@sha256:" not in image:
            raise ValueError(f"image {image!r} must be pinned by digest (repo@sha256:...)")
        return modal.Image.from_registry(image).entrypoint([]).env({"PYTHONUNBUFFERED": "1", IMAGE_ENV: image})
    if commit:
        image = (
            modal.Image.from_registry(stack.BASE_IMAGE)
            .run_commands(stack.DEPS_INSTALL)
            .run_commands(
                f"cd {stack.MILES_SRC} && git fetch --no-tags {stack.FORK_REPO} {commit} "
                f"&& git checkout --force --detach {commit} "
                "&& { [ ! -f fork/deps/sglang/apply.py ] "  # commits before SGLang patches
                f"|| python fork/deps/sglang/apply.py {stack.SGLANG_SRC} \"$(python -c '{_COMMIT_PIN}')\"; }} "
                "&& python fork/tools/ledger.py markers"
            )
            .env({"PYTHONPATH": str(stack.MILES_SRC), "MILES_MOBIUS_VERSION": f"commit@{commit[:10]}",
                  "MILES_MOBIUS_COMMIT": commit})
        )
    elif stack.FORK_IMAGE:
        image = modal.Image.from_registry(stack.FORK_IMAGE).env({IMAGE_ENV: stack.FORK_IMAGE})
    else:
        raise RuntimeError("miles_ext/stack.py: FORK_IMAGE is empty; publish it or pass a fork commit")
    return image.entrypoint([]).env({"PYTHONUNBUFFERED": "1"})

