"""Run the GLM-5.3-Flash GPU-delta benchmark on one Modal B300:8 node."""

import os
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
app = modal.App("miles-rl-gpudelta-glmbench")
weights = modal.Volume.from_name("miles-rl-weights")
results = modal.Volume.from_name("miles-rl-results")
if modal.is_local():
    from miles_ext import stack

    BASE_IMAGE, SGLANG_COMMIT, NVCOMP = stack.BASE_IMAGE, stack.SGLANG_COMMIT, stack.NVCOMP_VERSION
else:
    BASE_IMAGE = SGLANG_COMMIT = NVCOMP = "unused-in-container"
image = (
    modal.Image.from_registry(BASE_IMAGE, add_python=None)
    .entrypoint([])
    .run_commands(
        f"python3 -m pip install -q --no-deps --only-binary=:all: nvidia-libnvcomp-cu13=={NVCOMP} "
        "python-snappy==0.7.3 cramjam==2.11.0 lz4 zstandard"
    )
    .add_local_dir(ROOT / "fork/deps/sglang", "/p", copy=True)
    .run_commands(f"python3 /p/apply.py /sgl-workspace/sglang {SGLANG_COMMIT}")
    .add_local_dir(ROOT / "miles", "/root/miles/miles", ignore=["**/__pycache__"])
    .add_local_dir(ROOT / "tests/manual/gpu_delta", "/root/miles/tests/manual/gpu_delta", ignore=["**/__pycache__"])
    .add_local_file(Path(__file__).with_name("glm_bench.py"), "/root/glm_bench.py")
)


@app.function(
    image=image,
    gpu="B300:8",
    cpu=64,
    memory=524288,
    timeout=7200,
    ephemeral_disk=1048576,
    volumes={"/w": weights.read_only(), "/results": results},
)
def bench(tag: str, model: str, extra: list):
    out = f"/results/_probes/glm-bench-{tag}"
    os.makedirs(out, exist_ok=True)
    cmd = [
        "python3",
        "/root/glm_bench.py",
        "--model",
        model,
        "--output",
        "/tmp/bench/run",
        "--prepared-inputs",
        "/tmp/bench/inputs",
        "--codecs",
        "snappy-zstd",
        "--hashing",
        "off",
        "--versions",
        "3",
        "--ports",
        "31000",
        *extra,
    ]
    env = {**os.environ, "PYTHONPATH": "/root/miles:/sgl-workspace/sglang/python"}
    with open(f"{out}/bench.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd="/root/miles").returncode
    subprocess.run(
        f"cp -r /tmp/bench/run {out}/run; mkdir -p {out}/logs; "
        f"find /tmp/bench -name '*.log' -size -50M -exec cp --parents {{}} {out}/logs/ \; ; true",
        shell=True,
    )
    results.commit()
    return rc, open(f"{out}/bench.log").read()[-15000:]


@app.local_entrypoint()
def main(tag: str, model: str = "/w/gpudelta/GLM-5.3-Flash-fp8-4layer", validate: bool = True):
    rc, tail = bench.remote(tag, model, ["--validate-final-target"] if validate else [])
    print(tail)
    if rc:
        raise SystemExit(f"benchmark failed (exit {rc}); logs in miles-rl-results:/_probes/glm-bench-{tag}/")
