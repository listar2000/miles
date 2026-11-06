"""What every role of a node sets up the same way."""

import contextlib
import os
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from miles_ext.node import storage as node_storage
from miles_ext.node.contract import GPU_ENV
from miles_ext.platform import compile_cache

# as upstream's launchers do: the image's PYTHONPATH would hide packages like megatron.post_training
MEGATRON_SRC = "/root/Megatron-LM"
LOCAL_LOGS = Path("/tmp/miles-logs")  # a node's log while its results storage is not live
TEE_EXIT_SECS = 5


def pythonpath() -> str:
    return ":".join(p for p in (os.environ.get("PYTHONPATH"), MEGATRON_SRC) if p)


def compile_cache_key(gpu: str) -> str:
    return compile_cache.key(os.environ.get(GPU_ENV) or gpu, version("sglang"))


def restore_compile_cache(cache_key: str) -> None:
    """Unpack what earlier nodes saved. Neither step ever fails a run: a cold start is the fallback."""
    try:
        node_storage.for_mount(compile_cache.MOUNT).refresh()
    except Exception as e:
        print(f"compile cache: refresh failed: {type(e).__name__}: {e}", flush=True)
    compile_cache.restore(cache_key)


def _publish_log(path: Path, dest: Path, storage) -> None:
    """Add this segment's log to ``dest`` on the storage. A failure here never fails the node."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as src, dest.open("ab") as out:
            shutil.copyfileobj(src, out)
        storage.commit()
    except Exception as e:
        print(f"log {path} not published to {dest}: {type(e).__name__}: {e}", flush=True)


@contextlib.contextmanager
def node_log(results: Path, run_id: str, name: str, storage):
    """Copy this process's stdout and stderr, and its children's, to ``<results>/<run_id>/logs/<name>.log``.

    Live storage takes the file as it grows. Otherwise the file is written locally and added to the storage on exit,
    because an open file blocks a Modal volume's reload and a mirror would upload it on every commit."""
    dest = results / run_id / "logs" / f"{name}.log"
    path = dest if storage.live else LOCAL_LOGS / run_id / f"{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not storage.live:  # what an earlier segment left here is already on the storage
        path.unlink(missing_ok=True)
    stdout, stderr = os.dup(1), os.dup(2)
    tee = subprocess.Popen(["tee", "-a", str(path)], stdin=subprocess.PIPE, stdout=stdout, start_new_session=True)
    os.dup2(tee.stdin.fileno(), 1)
    os.dup2(tee.stdin.fileno(), 2)
    try:
        yield path
    finally:
        if not storage.live:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout, 1)
            os.dup2(stderr, 2)
            tee.stdin.close()
            with contextlib.suppress(subprocess.TimeoutExpired):  # a child that outlives the role holds the pipe
                tee.wait(TEE_EXIT_SECS)
            _publish_log(path, dest, storage)
