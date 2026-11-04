"""Measure Megatron checkpoint saves and uploads on one GPU node (H100:8 by default).

CKPT_PROBE_GPU picks another node (e.g. B300:1), CKPT_PROBE_IMAGE another image; ranks = the node's GPUs."""

import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

import modal

BASE_IMAGE = os.environ.get(
    "CKPT_PROBE_IMAGE", "radixark/miles:v0.1.2@sha256:c81d8ba7b91571aac8e0bb2be372c6a9664e1ff7b88f518a8891861bae83e3b7"
)
GPU = os.environ.get("CKPT_PROBE_GPU", "H100:8")
NPROC = int(GPU.split(":")[1]) if ":" in GPU else 1
VOL, LOCAL = "/vol", "/tmp/ckpt"
WORKER = "/root/ckpt_async_worker.py"

app = modal.App("miles-ckpt-probe-gpu")
vol = modal.Volume.from_name("miles-ckpt-probe", version=2)
image = (
    modal.Image.from_registry(BASE_IMAGE)
    .entrypoint([])
    .env({"PYTHONUNBUFFERED": "1", "CKPT_PROBE_GPU": GPU})
    .pip_install("py-spy")
    .add_local_file(Path(__file__).with_name("ckpt_async_worker.py"), WORKER)
)

# name: (target, gib per GPU, threads, async, repeat, verify, disk load, async strategy, persistent worker, timeout min)
CASES = {
    "sync-t1": (VOL, 8, 1, False, 1, False, False, None, False, 25),
    "sync-t8": (VOL, 8, 8, False, 1, False, False, None, False, 25),
    "sync-t1-local": (LOCAL, 8, 1, False, 1, False, False, None, False, 25),
    "sync-t1-64": (VOL, 64, 1, False, 1, False, False, None, False, 25),
    "sync-t1-64-local": (LOCAL, 64, 1, False, 1, False, False, None, False, 25),
    # GLM-5.3 full-param shard size (627 GB over 8 ranks); -hog first fills host RAM as the trainer does
    "sync-t1-78": (VOL, 78, 1, False, 1, False, False, None, False, 25),
    "sync-t1-78-hog": (VOL, 78, 1, False, 1, False, False, None, False, 25),
    "local-then-copy": (LOCAL, 64, 1, False, 1, False, False, None, False, 25),
    "local-then-copy-diskload": (LOCAL, 64, 1, False, 1, False, True, None, False, 25),
    "async-nvrx-persistent": (VOL, 8, 8, True, 2, True, False, "nvrx", True, 5),
    "async-mcore-persistent": (VOL, 8, 8, True, 2, True, False, "mcore", True, 5),
    "async-mcore-persistent-64": (VOL, 64, 8, True, 2, True, False, "mcore", True, 25),
    "async-nvrx-temporal": (VOL, 8, 8, True, 2, True, False, "nvrx", False, 5),
    "async-mcore-temporal": (VOL, 8, 8, True, 2, True, False, "mcore", False, 5),
}
HOG_GIB = {"sync-t1-78-hog": int(os.environ.get("CKPT_PROBE_HOG_GIB", "1700"))}  # host RAM held while saving
HOG = "import sys, time; b = bytearray(int(sys.argv[1]) << 30); b[::4096] = b'x' * len(b[::4096]); print('hog ready', flush=True); time.sleep(86400)"
COPY_JOBS = 16
DISK_LOAD = (
    "mkdir -p /tmp/optload; while true; do dd if=/dev/zero of=/tmp/optload/f bs=64M count=320 conv=fsync "
    "status=none; dd if=/tmp/optload/f of=/dev/null bs=64M status=none; done"
)


def _dump_and_kill(proc: subprocess.Popen) -> None:
    """Print where every rank is stuck (py-spy), then kill the case's whole process group."""
    subprocess.run(f"for p in $(pgrep -f {WORKER}); do py-spy dump --pid $p | head -60; done", shell=True)
    proc.timed_out = True
    os.killpg(proc.pid, signal.SIGKILL)


def _used_gib() -> tuple[float, float]:
    """(host RAM in use, /dev/shm in use), GiB."""
    info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
    kib = lambda k: int(info[k].split()[0])  # noqa: E731
    shm = shutil.disk_usage("/dev/shm")
    return (kib("MemTotal") - kib("MemAvailable")) / (1 << 20), shm.used / (1 << 30)


class _PeakRam:
    def __init__(self):
        self.peak, self.shm_peak, self._stop = 0.0, 0.0, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(1):
            ram, shm = _used_gib()
            self.peak, self.shm_peak = max(self.peak, ram), max(self.shm_peak, shm)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()


def _run_case(name: str) -> dict:
    target, gib, threads, use_async, repeat, verify, disk_load, strategy, persistent, minutes = CASES[name]
    out = f"{target}/{name}"
    shutil.rmtree(out, ignore_errors=True)
    cmd = (
        [
            "torchrun",
            "--nproc-per-node",
            str(NPROC),
            WORKER,
            "--case",
            name,
            "--out",
            out,
            "--gib",
            str(gib),
            "--threads",
            str(threads),
            "--repeat",
            str(repeat),
        ]
        + ["--async-save"] * use_async
        + ["--persistent-worker"] * persistent
        + ["--verify"] * verify
    )
    env = os.environ | ({"ASYNC_STRATEGY": strategy} if strategy else {})
    load = subprocess.Popen(["bash", "-c", DISK_LOAD]) if disk_load else None
    hog = None
    if name in HOG_GIB:
        hog = subprocess.Popen(["python", "-c", HOG, str(HOG_GIB[name])], stdout=subprocess.PIPE, text=True)
        hog.stdout.readline()
        print(f"hog holds {HOG_GIB[name]} GiB; host RAM used {_used_gib()[0]:.1f} GiB", flush=True)
    lines = []
    try:
        with _PeakRam() as ram:
            # A separate session lets the timeout kill every rank in the case.
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env=env, start_new_session=True)
            timer = threading.Timer(minutes * 60, _dump_and_kill, args=(proc,))
            timer.start()
            for ln in proc.stdout:
                print(ln, end="", flush=True)
                lines += [ln] if ln.startswith("CASE ") else []
            proc.wait()
            timer.cancel()
    finally:
        if hog:
            hog.kill()
        if load:
            load.kill()
            shutil.rmtree("/tmp/optload", ignore_errors=True)
    if proc.returncode or not lines:
        return {
            "case": name,
            "error": proc.returncode,
            "timed_out": getattr(proc, "timed_out", False),
            "peak_dev_shm_gib": round(ram.shm_peak, 1),
        }
    res = json.loads(lines[-1][5:]) | {
        "peak_host_ram_used_gib": round(ram.peak, 1),
        "peak_dev_shm_gib": round(ram.shm_peak, 1),
    }
    if name.startswith("local-then-copy"):
        t = time.monotonic()
        subprocess.run(
            f"cd {out} && find . -type d -exec mkdir -p {VOL}/{name}/{{}} \\; && find . -type f -print0 "
            f"| xargs -0 -P {COPY_JOBS} -I{{}} cp {{}} {VOL}/{name}/{{}}",
            shell=True,
            check=True,
        )
        res["copy_to_volume_s"] = round(time.monotonic() - t, 1)
        res["copy_gbs"] = round(NPROC * (gib << 30) / 1e9 / res["copy_to_volume_s"], 3)
    if target == VOL or name.startswith("local-then-copy"):
        t = time.monotonic()
        vol.commit()
        res["commit_s"] = round(time.monotonic() - t, 1)
    shutil.rmtree(out, ignore_errors=True)
    print("CASE " + json.dumps(res), flush=True)
    return res


@app.function(
    image=image,
    gpu=GPU,
    cpu=64,
    memory=int(os.environ.get("CKPT_PROBE_MEMORY_GIB", "1024")) * 1024,
    ephemeral_disk=3 * 1024 * 1024,
    volumes={VOL: vol},
    timeout=3 * 3600,
)
def probe(only: str) -> list[dict]:
    print(f"host RAM used at start: {_used_gib()[0]:.1f} GiB, /dev/shm {_used_gib()[1]:.1f} GiB", flush=True)
    rows = [_run_case(n) for n in (only.split(",") if only else CASES)]
    for n in CASES:
        shutil.rmtree(f"{VOL}/{n}", ignore_errors=True)
    vol.commit()
    return rows


@app.local_entrypoint()
def main(only: str = ""):
    print("SUMMARY " + json.dumps(probe.remote(only)), flush=True)
