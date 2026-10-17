"""Compiled-kernel caches that outlive the container: design in fork/patches/ext-008-compile-cache.md.

SGLang keeps FlashInfer autotune results and the DeepGEMM, Triton, inductor, TileLang and CUDA
kernel caches under ``SGLANG_CACHE_DIR``, which is container-local. This module keeps it on a
Volume as one tar file per node, ``<volume>/<key>/node-<rank>.tar``:

- ``restore`` unpacks every node's archive into the local cache, oldest save first, before Ray starts;
- each node's ``Saver`` packs its local cache back whenever it changed and settled, and on exit;
- ``env`` points the run's own Triton, inductor, TileLang and CUDA caches at the same directory.

A failed restore or save is logged and never fails a run.
"""

import os
import tarfile
import threading
import time
from pathlib import Path
from typing import Callable

LOCAL = Path("/root/.cache/sglang")  # SGLang's default SGLANG_CACHE_DIR
MOUNT = Path("/compile-cache")
SAVE_EVERY_SECS = 300
MIN_AGE_SECS = 60  # FlashInfer writes all ranks' tuning files together; wait until the burst is over


def key(gpu: str, sglang_version: str) -> str:
    return f"{gpu.lower()}/sglang-{sglang_version}"


def archive(cache_key: str, mount: Path = MOUNT, node: int = 0) -> Path:
    return mount / cache_key / f"node-{node}.tar"


def _archives(cache_key: str, mount: Path) -> list[Path]:
    """Oldest save first: every node saves what it restored too, so the newest archive's copy of
    a file is the newest one."""
    found = (mount / cache_key).glob("node-*.tar")
    return sorted((p for p in found if p.stem[5:].isdigit()), key=lambda p: (p.stat().st_mtime_ns, p.name))


def env(local: Path = LOCAL) -> dict[str, str]:
    """For every process of the run (Ray workers inherit it). SGLang points Triton, inductor,
    TileLang, CUDA and FlashInfer at ``SGLANG_CACHE_DIR`` with setdefault, so the trainer's
    kernels land in the same directory only if these are set here too."""
    return {"SGLANG_CACHE_DIR": str(local), "TRITON_CACHE_DIR": str(local / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(local / "inductor"), "TILELANG_CACHE_DIR": str(local / "tilelang"),
            "CUDA_CACHE_PATH": str(local / "nv"),
            # Miles' engine env defaults this to "1" (miles/ray/specs/inference.py), which puts DeepGEMM's
            # kernels in a pid_<pid> directory no later launch reads; one engine per node needs no split,
            # and several engines on one node rely on DeepGEMM's own locks/ directory
            "SGLANG_DG_CACHE_DIR_PER_PROCESS": "0"}


def _skipped(rel: Path) -> bool:
    return any(part == "locks" or part.endswith((".lock", ".tmp")) for part in rel.parts)


def _files(root: Path) -> list[Path]:
    out = []
    for dirpath, _, names in os.walk(root):
        for name in names:
            rel = (Path(dirpath) / name).relative_to(root)
            if not _skipped(rel) and not (root / rel).is_symlink():
                out.append(rel)
    return sorted(out)


def _state(root: Path, files: list[Path]) -> tuple:
    """What a save would contain: changes when any file is added, removed or rewritten."""
    return tuple((str(rel), (st := (root / rel).stat()).st_size, st.st_mtime_ns) for rel in files)


def restore(cache_key: str, mount: Path = MOUNT, local: Path = LOCAL, log=print) -> None:
    t = time.monotonic()
    try:
        srcs = _archives(cache_key, mount)
        if not srcs:
            log(f"compile cache: nothing saved under {cache_key} yet; compiling from scratch")
            return
        local.mkdir(parents=True, exist_ok=True)
        for src in srcs:
            with tarfile.open(src) as tar:
                tar.extractall(local, filter="data")  # plain files only, nothing outside ``local``
        size = sum(src.stat().st_size for src in srcs)
        log(f"compile cache: restored {size / 1e6:.0f} MB ({len(srcs)} nodes) from {cache_key} "
            f"in {time.monotonic() - t:.0f}s")
    except Exception as e:  # a cold start is the fallback, never a failed run
        log(f"compile cache: restore failed ({type(e).__name__}: {e}); compiling from scratch")


def save(cache_key: str, commit: Callable[[], None], mount: Path = MOUNT, local: Path = LOCAL, node: int = 0,
         min_age: float = MIN_AGE_SECS, last: tuple | None = None, log=print) -> tuple | None:
    """Pack this node's local cache into the volume unless it is unchanged since ``last`` or a file is
    younger than ``min_age``. Returns the state saved (or ``last`` when nothing was saved)."""
    try:
        files = _files(local) if local.is_dir() else []
        state = _state(local, files)
        if not files or state == last:
            return last
        newest = max(m for _, _, m in state) / 1e9
        if time.time() - newest < min_age:
            return last
        t = time.monotonic()
        dst = archive(cache_key, mount, node)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f"{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")  # one per saving thread
        with tarfile.open(tmp, "w") as tar:
            for rel in files:
                tar.add(local / rel, arcname=str(rel), recursive=False)
        os.replace(tmp, dst)  # readers see the old archive or the new one, never a partial one
        commit()
        log(f"compile cache: saved {len(files)} files ({dst.stat().st_size / 1e6:.0f} MB) to {cache_key}/{dst.name} "
            f"in {time.monotonic() - t:.0f}s")
        return state
    except Exception as e:
        log(f"compile cache: save failed ({type(e).__name__}: {e})")
        return last


class Saver:
    """A node's background save every ``every`` seconds; ``stop`` saves once more, whatever the age."""

    def __init__(self, cache_key: str, commit: Callable[[], None], every: float = SAVE_EVERY_SECS, **kw):
        self._args, self._kw, self._every = (cache_key, commit), kw, every
        self._last = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "Saver":
        local = self._kw.get("local", LOCAL)  # what restore brought in is on the volume already
        try:
            self._last = _state(local, _files(local)) if local.is_dir() else None
        except OSError:  # a file vanished mid-walk: the first save then repacks everything
            self._last = None
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self._every):
            self._last = save(*self._args, **self._kw, last=self._last)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=120)
        save(*self._args, **{**self._kw, "min_age": 0}, last=self._last)
