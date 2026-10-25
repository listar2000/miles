"""Trainer side of the split adapter sync (ext-037): send each adapter sync through the results volume."""

import json
import os
import shutil
import threading
import time
import urllib.request
from pathlib import Path

from miles_ext.launch.split.agent import MANIFEST, PUSH_TIMEOUT_SECS, TENSORS

AGENT_ENV = "MILES_SPLIT_AGENT_URL"  # the trainer's local relay to the agent
RUN_DIR_ENV = "MILES_SPLIT_RUN_DIR"  # /results/<run_id> on the trainer
COMMIT_DIR = Path("/tmp/miles-split-commit")  # sender -> function process: "commit this sync"
COMMIT_TIMEOUT_SECS = 300
KEEP_SYNCS = 2  # sync dirs kept on the volume (the newest; an engine may still be reading the last)


# ---- trainer side ---------------------------------------------------------------------------------


def _protocol_base():
    from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol

    return WeightTransferProtocol


class VolumeAdapterSync(_protocol_base()):
    supports_lora = True

    def __init__(self, args) -> None:
        super().__init__(args)
        self.group_name = "miles-split-volume"
        self._agent = os.environ[AGENT_ENV]
        self._run_dir = Path(os.environ[RUN_DIR_ENV])
        self._buckets: list[list] = []
        self._version: int | None = None

    def connect(
        self,
        rollout_engines,
        engine_gpu_counts,
        engine_gpu_offsets,
        parallel_state,
        placement,
        selector,
        engine_cell_ids=None,
    ) -> None:
        from miles.backends.training_utils.weight_update.protocols.utils.data_replica import (
            get_data_replica_rank_and_size,
        )

        self.rollout_engines = rollout_engines
        self._selector = selector
        if engine_cell_ids is not None:  # Miles v0.1.3+: the session frame runs through per-cell updaters
            from miles.backends.training_utils.weight_update.rollout_cell_updater import create_rollout_cell_updaters

            self.cell_updaters_of_cell_id = create_rollout_cell_updaters(self.args, rollout_engines, engine_cell_ids)
        # one sender per replica set, as the broadcast protocol picks it
        replica_rank, _ = get_data_replica_rank_and_size(parallel_state, placement)
        self.is_sender = replica_rank == 0
        if self.is_sender:
            _call(f"{self._agent}/connect", {"engines": len(rollout_engines)}, timeout=60)

    def begin_sync(self, weight_version, iter_buckets) -> bool:
        self._version, self._buckets = int(weight_version), []
        return True

    def send_bucket(self, bucket) -> None:
        self._buckets.append([(name, t.detach().to("cpu", copy=True).contiguous()) for name, t in bucket])
        bucket.clear()

    def after_base_weights(self) -> None:
        if not self.is_sender:
            return
        t0 = time.monotonic()
        sync_dir = write_sync(self._run_dir / "adapters", self._version, self._buckets)
        t1 = time.monotonic()
        request_commit(sync_dir.name)
        t2 = time.monotonic()
        _call(f"{self._agent}/push", {"dir": str(sync_dir), "selector": self._selector}, timeout=PUSH_TIMEOUT_SECS)
        t3 = time.monotonic()
        self._buckets = []
        prune(self._run_dir / "adapters", keep=KEEP_SYNCS)
        mb = (sync_dir / TENSORS).stat().st_size / 2**20
        self.update_weight_metrics.update(
            {
                "split/sync_write_s": t1 - t0,
                "split/sync_commit_s": t2 - t1,
                "split/sync_apply_s": t3 - t2,
                "split/sync_mb": mb,
            }
        )
        print(
            f"[split sync v{self._version}] {mb:.0f} MB: write {t1 - t0:.1f}s commit {t2 - t1:.1f}s "
            f"apply {t3 - t2:.1f}s",
            flush=True,
        )


def write_sync(root: Path, version: int, buckets: list[list]) -> Path:
    """Write one sync's tensors and bucket order to ``root/v<version>``."""
    from safetensors.torch import save_file

    out = root / f"v{version:07d}"
    tmp = root / f".v{version:07d}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    save_file({name: t for bucket in buckets for name, t in bucket}, str(tmp / TENSORS))
    (tmp / MANIFEST).write_text(json.dumps({"version": version, "buckets": [[n for n, _ in b] for b in buckets]}))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return out


def prune(root: Path, keep: int) -> None:
    for old in sorted(root.glob("v*"))[:-keep]:
        shutil.rmtree(old, ignore_errors=True)


def request_commit(name: str, timeout: float = COMMIT_TIMEOUT_SECS) -> None:
    """Ask the trainer's function process (``commit_watcher``) to commit the results volume, and wait."""
    COMMIT_DIR.mkdir(parents=True, exist_ok=True)
    req, done = COMMIT_DIR / f"{name}.req", COMMIT_DIR / f"{name}.done"
    done.unlink(missing_ok=True)
    req.write_text(str(time.time()))
    deadline = time.monotonic() + timeout
    while not done.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"results volume commit for {name} not done within {timeout}s")
        time.sleep(0.1)
    if (err := done.read_text()).strip():
        raise RuntimeError(f"results volume commit for {name} failed: {err}")


def commit_watcher(commit) -> threading.Event:
    """Commit the results volume for each sync request (run in the trainer's function process); returns its stop event."""
    stop = threading.Event()
    COMMIT_DIR.mkdir(parents=True, exist_ok=True)

    def loop():
        while not stop.is_set():
            for req in sorted(COMMIT_DIR.glob("*.req")):
                err = ""
                try:
                    commit()
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                req.unlink(missing_ok=True)
                tmp = req.with_suffix(".done.tmp")
                tmp.write_text(err)
                tmp.rename(req.with_suffix(".done"))  # the sender never reads a half-written result
            stop.wait(0.1)

    threading.Thread(target=loop, daemon=True).start()
    return stop


def _call(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read() or b"{}")
    if not out.get("ok"):
        raise RuntimeError(f"split agent {url}: {out}")
    return out
