"""Harbor's two per-trial task-dir hashes, computed off the event loop and shared by every process on the node
through a file cache (fork/patches/ext-001-task-hash.md)."""

import fcntl
import hashlib
import json
import logging
import os
import tempfile
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

# one entry per task dir: <sha of the path>.json, written under <sha>.lock (flock, so a killed holder frees it)
CACHE_DIR = Path(tempfile.gettempdir()) / "miles-task-hash"

# a dir listed on the node this recently is not listed again, so an in-place task edit can go unseen this long
FRESH_S = 300.0

# resolved task dir -> (signature digest, {kind: value}, time the files were listed); filled by warm, in a thread
_memo: dict[str, tuple[str, dict[str, Any], float]] = {}
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
    """Hash ``task_dir`` both ways unless its files (path, size, mtime) are unchanged since the last hash.

    Blocking (a directory listing, a full read when the dir is new or changed, or a wait for the process on the
    node doing either), so callers on an event loop run it in a thread; the trial's own lookups are then dict
    hits. Within ``FRESH_S`` of the last listing on the node it only reads the stored values.
    """
    root = Path(task_dir).resolve()
    with _guard:
        lock = _dir_locks.setdefault(str(root), threading.Lock())
    with lock:  # a second attempt of the same task waits for the first computation
        entry = _memo.get(str(root))
        if entry and time.time() - entry[2] < FRESH_S:
            return
        _memo[str(root)] = _shared_entry(root)


def _shared_entry(root: Path) -> tuple[str, dict[str, Any], float]:
    """The node's file-cache entry for ``root``: reused while fresh or while the files are unchanged, else Harbor's."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{hashlib.sha256(str(root).encode()).hexdigest()[:32]}.json"
    with open(path.with_suffix(".lock"), "a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)  # another process listing or hashing this dir: wait, then read
        try:
            stored = json.loads(path.read_text())
            if stored["root"] != str(root):
                stored = None
        except (OSError, ValueError, KeyError):
            stored = None
        now = time.time()
        if stored and now - stored["listed_at"] < FRESH_S:
            return stored["signature"], _values(stored), stored["listed_at"]
        digest = hashlib.sha256(repr(_signature(root)).encode()).hexdigest()
        if stored and stored["signature"] == digest:
            values = _values(stored)
        else:
            content_hash, files = _upstream_content_hash(root)
            values = {"content": (content_hash, files), "dirhash": _upstream_checksum(Task(root))}
        content_hash, files = values["content"]
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps(
                {
                    "root": str(root),
                    "signature": digest,
                    "listed_at": now,
                    "dirhash": values["dirhash"],
                    "content": [content_hash, [str(f) for f in files]],
                }
            )
        )
        os.replace(tmp, path)
        return digest, values, now


def _values(stored: dict[str, Any]) -> dict[str, Any]:
    content_hash, files = stored["content"]
    return {"content": (content_hash, [Path(f) for f in files]), "dirhash": stored["dirhash"]}


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
