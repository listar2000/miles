"""Probe which host memory kinds get a Modal container killed at an explicit memory limit."""

import os
import threading
import time

import modal

GPU_IMAGE = "radixark/miles:v0.1.2@sha256:c81d8ba7b91571aac8e0bb2be372c6a9664e1ff7b88f518a8891861bae83e3b7"
REQUEST_GIB = 16
LIMIT_GIB = 24  # the hard-limit cases: a container over this is OOM-killed
GIB = 1 << 30
CHUNK = 256 << 20

app = modal.App("miles-hostram-probe")
cpu_image = modal.Image.debian_slim().env({"PYTHONUNBUFFERED": "1"})
gpu_image = modal.Image.from_registry(GPU_IMAGE).entrypoint([]).env({"PYTHONUNBUFFERED": "1"})


def _meminfo() -> dict:
    out = {}
    for line in open("/proc/meminfo"):
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable", "AnonPages", "Cached", "Dirty", "Writeback", "Shmem", "Mlocked"):
            out[key] = round(int(value.split()[0]) / (1 << 20), 1)
    for name in ("memory.current", "memory.max", "memory.peak"):
        try:
            text = open(f"/sys/fs/cgroup/{name}").read().strip()
            out["cg_" + name.split(".")[1]] = round(int(text) / GIB, 1) if text.isdigit() else text
        except OSError:
            pass
    return out


def _sampler(case: str, stop: threading.Event) -> None:
    while not stop.wait(2):
        print(f"[{case}] {_meminfo()}", flush=True)


def _write_file(path: str, gib: int, drop: bool) -> None:
    buf = os.urandom(CHUNK)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        for i in range(gib * GIB // CHUNK):
            os.pwrite(fd, buf, i * CHUNK)
            if drop:
                os.fdatasync(fd)
                os.posix_fadvise(fd, i * CHUNK, CHUNK, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _read_file(path: str, drop: bool) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        offset = 0
        while chunk := os.pread(fd, CHUNK, offset):
            if drop:
                os.posix_fadvise(fd, offset, len(chunk), os.POSIX_FADV_DONTNEED)
            offset += len(chunk)
    finally:
        os.close(fd)


def run_case(case: str, gib: int) -> dict:
    stop = threading.Event()
    threading.Thread(target=_sampler, args=(case, stop), daemon=True).start()
    started, hold = time.time(), []
    print(f"[{case}] start, {gib} GiB against a {REQUEST_GIB} GiB request: {_meminfo()}", flush=True)
    if case in ("pagecache", "pagecache-drop"):
        drop = case.endswith("drop")
        _write_file("/tmp/probe.bin", gib, drop)
        print(f"[{case}] written: {_meminfo()}", flush=True)
        _read_file("/tmp/probe.bin", drop)
    elif case == "shm":
        _write_file("/dev/shm/probe.bin", gib, False)
    elif case == "anon":
        for _ in range(gib * GIB // CHUNK):
            hold.append(bytearray(os.urandom(1)) * CHUNK)
    elif case == "pinned":
        import torch

        for _ in range(gib):
            t = torch.ones(GIB, dtype=torch.uint8)
            assert int(torch.cuda.cudart().cudaHostRegister(t.data_ptr(), GIB, 0)) == 0
            hold.append(t)
    time.sleep(20)
    stop.set()
    result = {"case": case, "gib": gib, "survived": True, "seconds": round(time.time() - started), "end": _meminfo()}
    print(f"[{case}] {result}", flush=True)
    return result


@app.function(image=cpu_image, cpu=4, memory=REQUEST_GIB * 1024, timeout=1800)
def cpu_case(case: str, gib: int) -> dict:
    return run_case(case, gib)


@app.function(image=cpu_image, cpu=4, memory=(REQUEST_GIB * 1024, LIMIT_GIB * 1024), timeout=1800)
def cpu_case_limited(case: str, gib: int) -> dict:
    return run_case(case, gib)


@app.function(image=gpu_image, gpu="B300:1", cpu=8, memory=REQUEST_GIB * 1024, timeout=1800)
def gpu_case(case: str, gib: int) -> dict:
    return run_case(case, gib)


@app.function(image=gpu_image, gpu="B300:4", cpu=16, memory=2560 * 1024, timeout=600)
def b300x4_limit_case() -> dict:
    info = _meminfo()
    print(f"[b300x4] {info}", flush=True)
    return info


def _gather(calls: dict) -> None:
    for case, call in calls.items():
        try:
            print("RESULT", call.get())
        except Exception as error:  # a container killed for memory surfaces here
            print("RESULT", {"case": case, "survived": False, "error": f"{type(error).__name__}: {error}"[:400]})


@app.local_entrypoint()
def main(gib: int = 48):
    cases = ("pagecache", "pagecache-drop", "shm", "anon")
    _gather({case: cpu_case.spawn(case, gib if case != "anon" else REQUEST_GIB * 2) for case in cases})


@app.local_entrypoint()
def limited(gib: int = 48):
    cases = ("pagecache", "pagecache-drop", "shm", "anon")
    _gather({case: cpu_case_limited.spawn(case, gib if case != "anon" else REQUEST_GIB * 2) for case in cases})


@app.local_entrypoint()
def gpu(gib: int = 48):
    cases = ("pagecache", "pinned")
    _gather({case: gpu_case.spawn(case, gib if case == "pagecache" else REQUEST_GIB * 2) for case in cases})


@app.local_entrypoint()
def b300x4_limit():
    print("RESULT", b300x4_limit_case.remote())
