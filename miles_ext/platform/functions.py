"""The platform's Modal functions, in one app run on demand (``app.run``) in the target's environment.

- ``deep_check``: the validate step, in the fork image (CPU, or one GPU via ``with_options``);
- ``convert_fsdp``: FSDP's final save to HF weights, in the fork image (CPU);
- ``download``: base weights from the HF Hub, in the fork image too (CPU): importing
  ``miles_ext`` needs the ``miles`` tree, and the image is cached once built.

Volumes and secrets are per call (``with_options``), so nothing about a run is
fixed here. The image is ``stack.FORK_IMAGE``, ``MILES_IMAGE`` (a ``repo@sha256:``)
or the image built from a pushed commit when ``MILES_FORK_COMMIT`` is set when this
module is imported (the same switches as the test launcher); inside a container the
function's own image is used.
"""

import os
import subprocess

import modal

from miles_ext.platform.images import fork_image
from miles_ext.platform.target import Target, helper_app_name

app = modal.App("miles-platform")  # each run() names the app after its target (helper_app_name)
_fork = (fork_image(os.environ.get("MILES_FORK_COMMIT") or None, os.environ.get("MILES_IMAGE") or None)
         if modal.is_local() else modal.Image.debian_slim())


def run(target: Target):
    """``with functions.run(target):`` the helper app, in the target's environment, under its name."""
    return app.run(name=helper_app_name(target), environment_name=target.modal_environment)

CHECK_TIMEOUT = 1800
CONVERT_TIMEOUT = 2 * 3600
DOWNLOAD_TIMEOUT = 3 * 3600


@app.function(image=_fork, cpu=4, memory=16384, timeout=CHECK_TIMEOUT)
def deep_check(recipe: dict, run_id: str, libcuda_stub: bool = False) -> dict:
    from miles_ext.platform.validate import run_deep_check

    return run_deep_check(recipe, run_id, libcuda_stub=libcuda_stub)


@app.function(image=_fork, cpu=8, memory=98304, timeout=CONVERT_TIMEOUT)
def convert_fsdp(cmd: list[str]) -> str:
    subprocess.run(cmd, check=True)  # the results volume commits when the container exits
    return cmd[cmd.index("--output-dir") + 1]


@app.function(image=_fork, cpu=4, memory=8192, timeout=DOWNLOAD_TIMEOUT)
def download(hf_repo: str, revision: str, path: str) -> dict:
    from pathlib import Path

    from huggingface_hub import snapshot_download

    snapshot_download(hf_repo, revision=revision, local_dir=path, token=os.environ.get("HF_TOKEN") or None)
    return {"bytes": sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())}
