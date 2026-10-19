"""Harbor's two per-trial task-dir hashes, computed once per task in a worker thread instead of on the rollout loop.

``warm`` stores Harbor's own values; the patched ``Packager.compute_content_hash`` and ``Task.checksum`` read them
and fall back to Harbor for a dir ``warm`` never saw (fork/patches/ext-001-task-hash.md). ``prewarm`` hashes a
whole task set ahead of the first trials (agent_function.prepare, at rollout setup, hook-005).
"""

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from harbor.models.task.task import Task
from harbor.publisher.packager import Packager

logger = logging.getLogger(__name__)

_upstream_content_hash = Packager.compute_content_hash
_upstream_checksum = Task.checksum.fget

# resolved task dir -> (signature, {kind: value}); filled by warm, which runs in a thread
_memo: dict[str, tuple[tuple, dict[str, Any]]] = {}
_dir_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()


def _signature(task_dir: Path) -> tuple:
    out = []
    for p in sorted(task_dir.rglob("*")):
        if p.is_file():
            st = p.stat()
            out.append((p.relative_to(task_dir).as_posix(), st.st_size, st.st_mtime_ns))
    return tuple(out)


def _cached(kind: str, task_dir: Path | str, compute: Callable[[], Any]) -> Any:
    """The value warm stored for this dir, else Harbor's own computation (a dir warm never saw)."""
    entry = _memo.get(str(Path(task_dir).resolve()))
    if entry and kind in entry[1]:
        return entry[1][kind]
    return compute()


def _content_hash(task_dir: Path) -> tuple[str, list[Path]]:
    return _cached("content", task_dir, lambda: _upstream_content_hash(task_dir))


def _checksum(self: Task) -> str:
    return _cached("dirhash", self.task_dir, lambda: _upstream_checksum(self))


# Harbor's lock.py and trial.py look both up on the class at call time
Packager.compute_content_hash = staticmethod(_content_hash)
Task.checksum = property(_checksum)


def warm(task_dir: Path | str) -> None:
    """Hash ``task_dir`` both ways unless its files are unchanged since the last call (path, size, mtime).

    Blocking (a directory walk, and a full read when the dir is new or changed), so callers on an
    event loop run it in a thread; the trial's own lookups are then dict hits.
    """
    root = Path(task_dir).resolve()
    with _guard:
        lock = _dir_locks.setdefault(str(root), threading.Lock())
    with lock:  # a second attempt of the same task waits for the first computation
        sig = _signature(root)
        entry = _memo.get(str(root))
        if entry and entry[0] == sig:
            return
        values = {"content": _upstream_content_hash(root), "dirhash": _upstream_checksum(Task(root))}
        _memo[str(root)] = (sig, values)


def prewarm(task_dirs, workers: int = 4) -> None:
    """``warm`` every dir, in the given order, on ``workers`` threads of its own (not the default pool).

    A trial that needs a dir still being hashed waits on that dir's lock and then reads the stored values,
    so a task set that outlasts the call is never hashed twice. A dir that fails is left to the trial.
    """
    dirs = list(dict.fromkeys(str(Path(d).resolve()) for d in task_dirs))
    t0 = time.monotonic()

    def one(d: str) -> bool:
        try:
            warm(d)
            return True
        except Exception as e:
            logger.warning(f"task_hash prewarm of {d} failed, the trial hashes it: {e!r}")
            return False

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="task-hash-prewarm") as pool:
        done = sum(pool.map(one, dirs))
    logger.info(f"task_hash prewarm: {done}/{len(dirs)} task dirs hashed in {time.monotonic() - t0:.1f}s")
