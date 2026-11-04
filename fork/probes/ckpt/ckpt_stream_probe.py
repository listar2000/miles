"""Measure Modal checkpoint write and copy throughput with parallel streams (usage: fork/patches/ext-056-staged-save.md)."""

import json
import os
import shutil
import threading
import time

import modal

VOLUME = os.environ.get("CKPT_PROBE_VOLUME", "miles-ckpt-probe")
GPU = os.environ.get("CKPT_PROBE_GPU") or None
TOTAL_GIB = int(os.environ.get("CKPT_PROBE_GIB", "64"))
STREAMS = [int(s) for s in os.environ.get("CKPT_PROBE_STREAMS", "1,8,32,64").split(",")]
MNT = "/vol"
ROOT = f"{MNT}/ckpt-stream-probe/{(GPU or 'cpu').replace(':', 'x')}"
LOCAL = "/tmp/ckpt-stream-probe"
CHUNK = 64 << 20
VOL_ONLY = os.environ.get("CKPT_PROBE_VOL_ONLY") == "1"  # only the vol-N cases (a long sustained write)
WRITTEN = [0]  # bytes written by _raw so far; a ticker prints the rate every TICK seconds
TICK = 30

app = modal.App("miles-ckpt-stream-probe")
vol = modal.Volume.from_name(VOLUME, create_if_missing=True, version=2)
image = (
    modal.Image.debian_slim()
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .env({"CKPT_PROBE_VOLUME": VOLUME, "CKPT_PROBE_GPU": GPU or ""})
)


def _raw(path: str, gib: float, buf: memoryview) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        for _ in range(int(gib * (1 << 30)) // CHUNK):
            os.write(fd, buf)
            WRITTEN[0] += CHUNK
        os.fsync(fd)
    finally:
        os.close(fd)


def _ticker(stop: threading.Event) -> None:
    last, t0 = 0, time.monotonic()
    while not stop.wait(TICK):
        now = WRITTEN[0]
        print(
            json.dumps(
                {
                    "tick_s": round(time.monotonic() - t0),
                    "written_gib": now >> 30,
                    "gbs_last_tick": round((now - last) / 1e9 / TICK, 3),
                }
            ),
            flush=True,
        )
        last = now


def _range(path: str, gib: float, buf: memoryview, k: int) -> None:
    """One file written by k threads, each its own contiguous slice, with pwrite."""
    chunks = int(gib * (1 << 30)) // CHUNK
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd, chunks * CHUNK)

        def part(j: int) -> None:
            for c in range(j * chunks // k, (j + 1) * chunks // k):
                os.pwrite(fd, buf, c * CHUNK)

        _parallel(part, [(j,) for j in range(k)])
        os.fsync(fd)
    finally:
        os.close(fd)


def _torch(path: str, gib: float, ten) -> None:
    import torch

    with open(path, "wb") as f:
        for _ in range(int(gib * (1 << 30)) // ten.nbytes):
            torch.save(ten, f)
        f.flush()
        os.fsync(f.fileno())


def _parallel(fn, args_list) -> float:
    t = time.monotonic()
    threads = [threading.Thread(target=fn, args=a) for a in args_list]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return time.monotonic() - t


def _row(case: str, n: int, gib: float, secs: float, commit_s: float | None = None) -> dict:
    gb = gib * (1 << 30) / 1e9
    r = {"case": case, "streams": n, "gib": gib, "secs": round(secs, 1), "gbs": round(gb / secs, 3)}
    if commit_s is not None:
        r["commit_s"] = round(commit_s, 1)
        r["gbs_incl_commit"] = round(gb / (secs + commit_s), 3)
    print(json.dumps(r), flush=True)
    return r


@app.function(
    image=image,
    gpu=GPU,
    volumes={MNT: vol},
    cpu=32,
    memory=64 * 1024,
    ephemeral_disk=1024 * 1024,
    timeout=3 * 3600,
)
def probe(total_gib: int, streams: list[int]) -> list[dict]:
    import torch

    buf = memoryview(os.urandom(CHUNK))
    ten = torch.randint(0, 2**15, (128 << 20,), dtype=torch.int16)  # 256 MiB
    rows = [{"case": "host", "nproc": os.cpu_count(), "gpu": GPU}]
    print(json.dumps(rows[0]), flush=True)

    def vol_case(name: str, n: int, fn, arg) -> None:
        out = f"{ROOT}/{name}"
        os.makedirs(out, exist_ok=True)
        per = total_gib / n
        secs = _parallel(fn, [(f"{out}/s{i}.bin", per, arg) for i in range(n)])
        t = time.monotonic()
        vol.commit()
        rows.append(_row(name, n, total_gib, secs, time.monotonic() - t))
        shutil.rmtree(out, ignore_errors=True)
        vol.commit()

    stop = threading.Event()
    threading.Thread(target=_ticker, args=(stop,), daemon=True).start()
    for n in streams:
        vol_case(f"vol-{n}", n, _raw, buf)
    stop.set()
    if VOL_ONLY:
        return rows
    for n in (1, 8):
        vol_case(f"torch-{n}", n, _torch, ten)
    for k in (4, 8):
        out = f"{ROOT}/range-8x{k}"
        os.makedirs(out, exist_ok=True)
        secs = _parallel(_range, [(f"{out}/s{i}.bin", total_gib / 8, buf, k) for i in range(8)])
        t = time.monotonic()
        vol.commit()
        rows.append(_row(f"range-8x{k}", 8 * k, total_gib, secs, time.monotonic() - t))
        shutil.rmtree(out, ignore_errors=True)
        vol.commit()
    for n in sorted({8, max(streams)}):
        os.makedirs(LOCAL, exist_ok=True)
        per = total_gib / n
        src = [f"{LOCAL}/s{i}.bin" for i in range(n)]
        rows.append(_row(f"local-{n}", n, total_gib, _parallel(_raw, [(p, per, buf) for p in src])))
        out = f"{ROOT}/copy-{n}"
        os.makedirs(out, exist_ok=True)
        secs = _parallel(shutil.copyfile, [(p, f"{out}/s{i}.bin") for i, p in enumerate(src)])
        t = time.monotonic()
        vol.commit()
        rows.append(_row(f"copy-{n}", n, total_gib, secs, time.monotonic() - t))
        shutil.rmtree(out, ignore_errors=True)
        shutil.rmtree(LOCAL, ignore_errors=True)
        vol.commit()
    return rows


@app.local_entrypoint()
def main():
    rows = probe.remote(TOTAL_GIB, STREAMS)
    print("SUMMARY " + json.dumps(rows), flush=True)
