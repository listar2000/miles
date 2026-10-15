"""Images the platform functions run in."""

import modal

from miles_ext import stack

IMAGE_ENV = "MILES_EXT_IMAGE"  # the registry image a container runs (unset when Modal built it)


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
            .run_commands(f"cd {stack.MILES_SRC} && git fetch --no-tags {stack.FORK_REPO} {commit} "
                          f"&& git checkout --force --detach {commit} && python fork/tools/ledger.py markers")
            .env({"PYTHONPATH": str(stack.MILES_SRC), "MILES_FORK_RELEASE": f"commit-{commit[:10]}",
                  "MILES_FORK_COMMIT": commit})
        )
    elif stack.FORK_IMAGE:
        image = modal.Image.from_registry(stack.FORK_IMAGE).env({IMAGE_ENV: stack.FORK_IMAGE})
    else:
        raise RuntimeError("miles_ext/stack.py: FORK_IMAGE is empty; publish it or pass a fork commit")
    return image.entrypoint([]).env({"PYTHONUNBUFFERED": "1"})

