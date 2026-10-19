"""Stage Megatron checkpoints on node-local disk and upload them in the background (patch page ext-031)."""

import contextlib
import os
import re
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STAGE_ENV = "MILES_EXT_SAVE_STAGE"
COPY_THREADS = 16
UPLOAD_TIMEOUT_SECS = 1800  # how long exit waits for pending uploads
POLL_SECS = 2
SETTLE_SECS = 5  # markers of one save land within the barrier's skew; wait this long for stragglers
ITER = re.compile(r"iter_(\d{7})$")
MARKER = ".saved-rank"


def report_partition(run_id: str) -> str:
    return f"upload-{run_id}"


def _staged(stage: Path) -> list[Path]:
    return sorted(p for p in stage.glob("iter_*") if ITER.search(p.name))


def _marked(stage: Path) -> list[Path]:
    """Staged saves that finished (an unmarked one is a save cut short)."""
    return [it for it in _staged(stage) if any(it.glob(f"{MARKER}*"))]


def _print(msg: str) -> None:
    print(msg, flush=True)


@contextlib.contextmanager
def context(args, iteration: int):
    """``--custom-megatron-save-context-path``: this rank's save goes to the node's stage."""
    import torch.distributed as dist

    stage = Path(os.environ[STAGE_ENV])
    target = stage / f"iter_{iteration:07d}"
    waited = time.monotonic()
    while [p for p in _staged(stage) if p != target]:  # another rank of this node may have made target already
        time.sleep(POLL_SECS)
    if time.monotonic() - waited > POLL_SECS:
        _print(f"staged save: iteration {iteration} waited {time.monotonic() - waited:.0f}s for the last upload")
    target.mkdir(parents=True, exist_ok=True)  # Megatron creates it on rank 0 only, which may be another node
    saved_to, args.save = args.save, str(stage)
    try:
        yield
    finally:
        args.save = saved_to
    (target / f"{MARKER}{dist.get_rank()}").touch()


class Uploader:
    """One node's background upload of staged saves. Rank 0's node (``node`` 0, where the
    supervisor watches the tracker) also collects every node's reports and writes the tracker."""

    def __init__(
        self,
        stage: str | Path,
        run_dir: str | Path,
        run_id: str,
        node: int,
        queue,
        commit: Callable[[], None],
        *,
        train_ranks: int,
        log=_print,
    ):
        from miles_ext.platform.layout import TRACKER  # trainer ranks import this module without the platform

        self.stage, self.ckpt_dir, self.tracker = Path(stage), Path(run_dir) / "checkpoints", TRACKER
        self.run_id, self.node, self.queue, self.commit = run_id, node, queue, commit
        self.train_ranks, self.log = train_ranks, log
        self.reports: dict[int, dict[int, int]] = {}  # node 0: iteration -> node -> ranks uploaded
        self.tracked = -1  # node 0: the last iteration it wrote to the tracker
        self.stop = threading.Event()
        self.stage.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._loop, daemon=True, name="staged-save-upload")

    def start(self) -> "Uploader":
        self.thread.start()
        return self

    def _ready(self) -> Path | None:
        for it in _marked(self.stage):
            markers = list(it.glob(f"{MARKER}*"))
            if time.time() - max(m.stat().st_mtime for m in markers) > SETTLE_SECS:
                return it
        return None

    def _upload(self, it: Path) -> None:
        n, began = int(ITER.search(it.name).group(1)), time.monotonic()
        ranks = len(list(it.glob(f"{MARKER}*")))
        files = [p for p in it.rglob("*") if p.is_file() and not p.name.startswith(MARKER)]
        dest = self.ckpt_dir / it.name

        def copy(src: Path) -> int:
            out = dest / src.relative_to(it)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, out)
            return src.stat().st_size

        with ThreadPoolExecutor(COPY_THREADS) as pool:
            size = sum(pool.map(copy, files))
        self.commit()
        self.queue.put((self.node, n, ranks), partition=report_partition(self.run_id))
        shutil.rmtree(it)
        secs = time.monotonic() - began
        self.log(
            f"staged save: node {self.node} uploaded iteration {n} ({ranks} ranks, {size / 2**30:.1f} GiB) "
            f"in {secs:.0f}s ({size / 2**30 / max(secs, 1e-9):.2f} GiB/s)"
        )

    def _collect(self) -> None:
        while (item := self.queue.get(partition=report_partition(self.run_id), block=False)) is not None:
            node, n, ranks = item  # Modal's non-blocking get returns None when the partition is empty
            self.reports.setdefault(n, {})[node] = ranks  # a retried upload reports its node again

    def _track(self) -> None:
        for n in sorted(self.reports):
            if n > self.tracked and sum(self.reports[n].values()) >= self.train_ranks:
                (self.ckpt_dir / self.tracker).write_text(str(n))
                self.tracked = n
                self.log(f"staged save: iteration {n} complete on the volume; tracker -> {n}")

    def busy(self) -> bool:
        """A finished save on this node is not uploaded yet, or (node 0) not in the tracker yet."""
        return bool(_marked(self.stage)) or any(n > self.tracked for n in self.reports)

    def _step(self) -> None:
        it = self._ready()
        if it is not None:
            self._upload(it)
        if self.node == 0:
            self._collect()
            self._track()

    def _loop(self) -> None:
        while not self.stop.is_set():
            try:
                self._step()
            except Exception as e:  # a failed upload leaves the stage as is; the next pass retries
                self.log(f"staged save: node {self.node} upload pass failed ({type(e).__name__}: {e})")
            self.stop.wait(POLL_SECS)

    def wait(self, timeout: float) -> bool:
        """Wait (the thread keeps uploading) until nothing is pending; True when nothing is."""
        by = time.monotonic() + timeout
        while self.busy():
            if time.monotonic() > by:
                self.log(f"staged save: node {self.node} still uploading after {timeout:.0f}s")
                return False
            time.sleep(POLL_SECS)
        return True

    def drain(self, timeout: float) -> bool:
        """Stop the thread, then upload what is left here. True when nothing is left.

        If the thread is still copying after ``timeout``, returns False and leaves that copy to it;
        a copy started here runs to the end before the deadline is checked again."""
        by = time.monotonic() + timeout
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout)
        if self.thread.is_alive():  # still copying: a second copier would race it
            self.log(f"staged save: node {self.node} drain timed out during an upload")
            return False
        while True:
            try:
                self._step()
            except Exception as e:
                self.log(f"staged save: drain pass failed ({type(e).__name__}: {e})")
            if not self.busy():
                return True
            if time.monotonic() > by:
                self.log(
                    f"staged save: node {self.node} drain timed out; staged {[p.name for p in _marked(self.stage)]}, "
                    f"untracked {sorted(n for n in self.reports if n > self.tracked)}"
                )
                return False
            time.sleep(POLL_SECS)
