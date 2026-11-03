"""How large a HW-decompress-capable host arena (port-001 HostAllocation, fix-074 HOST fallback) can a Modal
B300 container map? The full GLM receiver fails cuMemSetAccess with OUT_OF_MEMORY even for a ~0.5 GB/rank delta."""

import subprocess

import modal

from miles_ext.gpu_delta.run_glm_bench import image

app = modal.App("miles-rl-gpudelta-allocprobe")

PROBE = r'''
import sys, time
sys.path.insert(0, "/sgl-workspace/sglang/python")
import torch
torch.cuda.init()
from sglang.srt.weight_sync.gpu_delta import memory as m
MB = 1 << 20
for size in [1*MB, 16*MB, 64*MB, 128*MB, 256*MB, 512*MB, 1024*MB, 2048*MB, 4096*MB, 16384*MB]:
    try:
        a = m.HostAllocation(size, 0)
        print("OK", size // MB, "MiB capacity", a.capacity // MB, flush=True)
        a.close()
    except Exception as e:
        print("FAIL", size // MB, "MiB:", e, flush=True)
held = []
for i in range(64):
    try:
        held.append(m.HostAllocation(256*MB, 0))
    except Exception as e:
        print("cumulative 256MiB allocations failed at", i, e, flush=True)
        break
else:
    print("cumulative 64 x 256 MiB OK", flush=True)
for a in held:
    a.close()
'''


@app.function(image=image, gpu="B300:1", cpu=8, memory=262144, timeout=1200)
def probe():
    out = subprocess.run(["python3", "-c", PROBE], capture_output=True, text=True)
    return out.stdout + out.stderr[-4000:]


@app.local_entrypoint()
def main():
    print(probe.remote())
