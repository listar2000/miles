"""Host memory breakdown of each trainer rank and its node, logged once a minute with the minute's peak [ext-057].

Install with ``--custom-megatron-init-path miles_ext.obs.host_ram.install``. A daemon thread in every Megatron rank
samples every 5 s and logs one line a minute: the rank's own resident memory (anonymous, file, shared) and pinned bytes
held by PyTorch's host allocator; local rank 0 adds the node's view (meminfo, the cgroup's memory files when present,
``/dev/shm`` and local-disk use). Each value is the peak over the minute, so a save or a sync between lines still
shows. Read-only: nothing here allocates host memory beyond the thread.
"""

import logging
import os
import threading
import time

import torch

logger = logging.getLogger(__name__)

SAMPLE_S = 5
LOG_S = 60
GIB = 1024**3
_PROC_STATUS_KEYS = ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmLck")
_MEMINFO_KEYS = ("MemTotal", "MemAvailable", "AnonPages", "Cached", "Dirty", "Writeback", "Shmem", "Mlocked")
_CGROUP_STAT_KEYS = ("anon", "file", "file_dirty", "file_writeback", "shmem")
_DISK_PATHS = {"shm": "/dev/shm", "tmp": "/tmp"}
_started = False


def parse_kb_lines(text: str, keys: tuple[str, ...]) -> dict[str, int]:
    """``Key:  123 kB`` lines (/proc/self/status, /proc/meminfo) to bytes, for the keys present."""
    found = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        if name in keys and rest.split():
            found[name] = int(rest.split()[0]) * 1024
    return found


def parse_cgroup_stat(text: str) -> dict[str, int]:
    """cgroup v2 ``memory.stat`` (bytes) for the keys that split anonymous, page cache and shared memory."""
    found = {}
    for line in text.splitlines():
        name, _, value = line.partition(" ")
        if name in _CGROUP_STAT_KEYS:
            found[f"cg_{name}"] = int(value)
    return found


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def rank_sample() -> dict[str, int]:
    sample = parse_kb_lines(_read("/proc/self/status") or "", _PROC_STATUS_KEYS)
    try:
        sample["torch_pinned"] = torch.cuda.host_memory_stats().get("allocated_bytes.all.current", 0)
    except (AttributeError, RuntimeError):
        pass
    return sample


def node_sample() -> dict[str, int]:
    sample = parse_kb_lines(_read("/proc/meminfo") or "", _MEMINFO_KEYS)
    for name in ("memory.current", "memory.peak"):
        if (text := _read(f"/sys/fs/cgroup/{name}")) and text.strip().isdigit():
            sample[f"cg_{name.split('.')[1]}"] = int(text)
    sample.update(parse_cgroup_stat(_read("/sys/fs/cgroup/memory.stat") or ""))
    for label, path in _DISK_PATHS.items():
        try:
            st = os.statvfs(path)
            sample[f"{label}_used"] = (st.f_blocks - st.f_bfree) * st.f_frsize
        except OSError:
            pass
    return sample


def merge_peak(peak: dict[str, int], sample: dict[str, int]) -> dict[str, int]:
    return {key: max(peak.get(key, 0), sample.get(key, 0)) for key in peak.keys() | sample.keys()}


def format_line(rank: int, peak: dict[str, int]) -> str:
    fields = " ".join(f"{key}={value / GIB:.1f}" for key, value in sorted(peak.items()))
    return f"[host ram] rank={rank} peak over {LOG_S}s (GiB): {fields}"


def _loop(rank: int, with_node: bool) -> None:
    peak, last_log = {}, time.monotonic()
    while True:
        try:
            peak = merge_peak(peak, rank_sample() | (node_sample() if with_node else {}))
            if time.monotonic() - last_log >= LOG_S:
                logger.info(format_line(rank, peak))
                peak, last_log = {}, time.monotonic()
        except Exception:
            logger.exception("host ram sampler stopped")
            return
        time.sleep(SAMPLE_S)


def install(args=None) -> None:
    """``--custom-megatron-init-path`` entry point; idempotent. Never fails the run."""
    global _started
    if _started:
        return
    _started = True
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1)))
    threading.Thread(target=_loop, args=(rank, local_rank == 0), name="host-ram-sampler", daemon=True).start()
    logger.info("host memory breakdown logged every %d s (ext-057)", LOG_S)
