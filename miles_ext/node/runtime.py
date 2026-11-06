"""What every role of a node sets up the same way."""

import os
import subprocess
from importlib.metadata import version
from pathlib import Path

from miles_ext.node.contract import GPU_ENV
from miles_ext.platform import compile_cache

# as upstream's launchers do: the image's PYTHONPATH would hide packages like megatron.post_training
MEGATRON_SRC = "/root/Megatron-LM"


def pythonpath() -> str:
    return ":".join(p for p in (os.environ.get("PYTHONPATH"), MEGATRON_SRC) if p)


def compile_cache_key(gpu: str) -> str:
    return compile_cache.key(os.environ.get(GPU_ENV) or gpu, version("sglang"))


def tee_logs(results: Path, run_id: str, name: str) -> Path:
    """Copy this process's stdout and stderr, and its children's, to ``<results>/<run_id>/logs/<name>.log``, so a viewer
    reads a node's log from shared storage. Off Modal only: an open file there would block a volume's reload."""
    path = results / run_id / "logs" / f"{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    tee = subprocess.Popen(["tee", "-a", str(path)], stdin=subprocess.PIPE, stdout=os.dup(1), start_new_session=True)
    os.dup2(tee.stdin.fileno(), 1)
    os.dup2(tee.stdin.fileno(), 2)
    return path
