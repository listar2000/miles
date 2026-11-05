"""Time small file operations on a Modal volume while a large write runs on it, on a second volume and on local disk."""

import json
import multiprocessing as mp
import os
import shutil
import statistics
import threading
import time

import modal

VOLUME_A, VOLUME_B = "miles-ckpt-probe", "miles-ckpt-probe-b"
GPU = os.environ.get("CKPT_PROBE_GPU") or None
TOTAL_GIB = int(os.environ.get("CKPT_PROBE_GIB", "400"))
WRITERS = 8
CHUNK = 64 << 20
BASELINE_SECS = 60
PROBE_EVERY_SECS = 2
A, B, LOCAL = "/vol-a", "/vol-b", "/tmp/contention"

app = modal.App("miles-ckpt-contention-probe")
vol_a = modal.Volume.from_name(VOLUME_A, create_if_missing=True, version=2)
vol_b = modal.Volume.from_name(VOLUME_B, create_if_missing=True, version=2)
image = modal.Image.debian_slim()


def _writer(path: str, gib: float, written) -> None:
    buf = os.urandom(CHUNK)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    for _ in range(int(gib * (1 << 30)) // CHUNK):
        os.write(fd, buf)
        with written.get_lock():
            written.value += CHUNK
    os.fsync(fd)
    os.close(fd)


def _prober(name: str, root: str, stop: threading.Event, rows: list) -> None:
    """One thread per target, so a stall on one target does not delay the others' samples."""
    os.makedirs(root, exist_ok=True)
    path = f"{root}/events.jsonl"
    while not stop.is_set():
        began = time.time()
        with open(path, "a") as f:
            f.write("x" * 200 + "\n")
        appended = time.time()
        with open(path, "a") as f:
            f.write("y" * 200 + "\n")
            f.flush()
            os.fsync(f.fileno())
        synced = time.time()
        with open(path) as f:
            f.read(4096)
        read = time.time()
        rows.append((began, name, appended - began, synced - appended, read - synced))
        stop.wait(PROBE_EVERY_SECS)


def _summary(rows: list, name: str, start: float, end: float) -> dict:
    picked = [r for r in rows if r[1] == name and start <= r[0] < end]
    out = {"target": name, "samples": len(picked)}
    for i, op in ((2, "append"), (3, "fsync"), (4, "read")):
        values = sorted(r[i] for r in picked) or [0.0]
        out[f"{op}_p50_s"] = round(statistics.median(values), 3)
        out[f"{op}_max_s"] = round(values[-1], 3)
    return out


@app.function(
    image=image,
    gpu=GPU,
    cpu=32,
    memory=64 * 1024,
    ephemeral_disk=1024 * 1024,
    volumes={A: vol_a, B: vol_b},
    timeout=1500,
)
def probe(total_gib: int) -> dict:
    out_dir = f"{A}/big"
    os.makedirs(out_dir, exist_ok=True)
    rows: list = []
    stop = threading.Event()
    threads = [
        threading.Thread(target=_prober, args=(n, r, stop, rows), daemon=True)
        for n, r in (("same_volume", f"{A}/small"), ("other_volume", f"{B}/small"), ("local_disk", LOCAL))
    ]
    for t in threads:
        t.start()
    time.sleep(BASELINE_SECS)
    ctx = mp.get_context("spawn")
    written = ctx.Value("q", 0)
    writers = [
        ctx.Process(target=_writer, args=(f"{out_dir}/s{i}.bin", total_gib / WRITERS, written)) for i in range(WRITERS)
    ]
    write_start = time.time()
    for p in writers:
        p.start()
    while any(p.is_alive() for p in writers):
        time.sleep(30)
        print(json.dumps({"t_s": round(time.time() - write_start), "written_gib": written.value >> 30}), flush=True)
    for p in writers:
        p.join()
    write_end = time.time()
    time.sleep(BASELINE_SECS)
    after_end = time.time()
    stop.set()
    for t in threads:
        t.join(timeout=300)
    names = ("same_volume", "other_volume", "local_disk")
    result = {
        "gpu": GPU,
        "write_secs": round(write_end - write_start),
        "write_gbs": round(total_gib * (1 << 30) / 1e9 / (write_end - write_start), 3),
        "before": [_summary(rows, n, 0, write_start) for n in names],
        "during": [_summary(rows, n, write_start, write_end) for n in names],
        "after": [_summary(rows, n, write_end, after_end) for n in names],
    }
    shutil.rmtree(out_dir, ignore_errors=True)
    vol_a.commit()
    print("RESULT " + json.dumps(result), flush=True)
    return result


@app.local_entrypoint()
def main():
    probe.remote(TOTAL_GIB)
