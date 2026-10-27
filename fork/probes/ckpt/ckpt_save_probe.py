"""Measure checkpoint-sized writes and copies to a Modal volume (or a GCS bucket mount) from CPU containers."""

import json
import os
import shutil
import threading
import time

import modal

VOLUME = os.environ.get("CKPT_PROBE_VOLUME", "miles-ckpt-probe")  # another of ours, e.g. miles-fork-results
GCS_BUCKET = os.environ.get("CKPT_PROBE_GCS_BUCKET")  # write to this GCS bucket (Modal secret miles-gcs-hmac) instead
MNT = "/vol"
ROOT = f"{MNT}/ckpt-probe"  # everything this probe writes, removed per case
CHUNK = 64 << 20

app = modal.App("miles-ckpt-probe-cpu")
vol = modal.Volume.from_name(VOLUME)
store = (
    modal.CloudBucketMount(
        GCS_BUCKET,
        bucket_endpoint_url="https://storage.googleapis.com",
        secret=modal.Secret.from_name("miles-gcs-hmac"),
    )
    if GCS_BUCKET
    else vol
)
image = modal.Image.debian_slim().env({"CKPT_PROBE_VOLUME": VOLUME, "CKPT_PROBE_GCS_BUCKET": GCS_BUCKET or ""})


def _commit() -> None:
    if not GCS_BUCKET:  # a bucket mount has nothing to commit: each file is uploaded when it is closed
        vol.commit()


def _write_file(path: str, gib: int, buf: memoryview) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        for _ in range((gib << 30) // CHUNK):
            os.write(fd, buf)
        os.fsync(fd)
    finally:
        os.close(fd)


def _parallel(fn, args_list) -> float:
    t = time.monotonic()
    threads = [threading.Thread(target=fn, args=a) for a in args_list]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return time.monotonic() - t


@app.function(
    image=image, volumes={MNT: store}, cpu=16, memory=32 * 1024, ephemeral_disk=1024 * 1024, timeout=2 * 3600
)
def write_case(case: str, idx: int, streams: int, file_gb: int, shared_dir: bool, from_local: bool) -> dict:
    buf = memoryview(os.urandom(CHUNK))
    out = f"{ROOT}/{case}" if shared_dir else f"{ROOT}/{case}/c{idx}"
    os.makedirs(out, exist_ok=True)
    files = [f"{out}/c{idx}_s{s}.bin" for s in range(streams)]
    res = {
        "case": case,
        "container": idx,
        "streams": streams,
        "file_gb": file_gb,
        "shared_dir": shared_dir,
        "from_local": from_local,
        "gib": streams * file_gb,
        "nproc": os.cpu_count(),
        "target": f"gs://{GCS_BUCKET}" if GCS_BUCKET else VOLUME,
    }
    if from_local:
        src = [f"/tmp/ckpt-src/s{s}.bin" for s in range(streams)]
        os.makedirs("/tmp/ckpt-src", exist_ok=True)
        res["local_write_s"] = round(_parallel(_write_file, [(p, file_gb, buf) for p in src]), 1)
        res["write_s"] = round(_parallel(shutil.copyfile, list(zip(src, files, strict=True))), 1)
    else:
        res["write_s"] = round(_parallel(_write_file, [(p, file_gb, buf) for p in files]), 1)
    t = time.monotonic()
    _commit()
    res["commit_s"] = round(time.monotonic() - t, 1)
    res["write_gbs"] = round(res["gib"] * (1 << 30) / 1e9 / res["write_s"], 3)
    res["end_to_end_gbs"] = round(res["gib"] * (1 << 30) / 1e9 / (res["write_s"] + res["commit_s"]), 3)
    print(json.dumps(res), flush=True)
    return res


@app.function(image=image, volumes={MNT: store}, cpu=2, timeout=1800)
def clean(case: str) -> None:
    if not GCS_BUCKET:
        vol.reload()  # a fresh container sees the writers' files only after a reload
    shutil.rmtree(f"{ROOT}/{case}", ignore_errors=True)
    _commit()


def _grid(quick: bool) -> list[tuple[str, int, int, int, bool, bool]]:
    """(case, containers, streams, file_gb, shared_dir, from_local); ~32-64 GiB per container."""
    if quick:
        return [("q-s1", 1, 1, 8, False, False), ("q-s16", 1, 16, 2, False, False), ("q-c4", 4, 8, 4, False, False)]
    g = [(f"streams{s}", 1, s, max(64 // s, 1), False, False) for s in (1, 2, 4, 8, 16, 32)]
    g += [("bigfile", 1, 1, 64, False, False), ("smallfiles", 1, 32, 1, False, False)]
    g += [(f"containers{c}", c, 8, 4, False, False) for c in (2, 4, 8)]
    g += [("containers8-shared-dir", 8, 8, 4, True, False), ("local-to-volume", 1, 8, 4, False, True)]
    return g


@app.local_entrypoint()
def main(quick: bool = False):
    rows = []
    for case, containers, streams, file_gb, shared, local in _grid(quick):
        args = [(case, i, streams, file_gb, shared, local) for i in range(containers)]
        t = time.monotonic()
        parts = list(write_case.starmap(args))
        wall = time.monotonic() - t
        gib = sum(p["gib"] for p in parts)
        rows.append(
            {
                "case": case,
                "containers": containers,
                "streams": streams,
                "file_gb": file_gb,
                "gib": gib,
                "per_container_write_gbs": [p["write_gbs"] for p in parts],
                "max_commit_s": max(p["commit_s"] for p in parts),
                "aggregate_gbs_incl_startup": round(gib * (1 << 30) / 1e9 / wall, 3),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
        clean.remote(case)
    print("SUMMARY " + json.dumps(rows), flush=True)
