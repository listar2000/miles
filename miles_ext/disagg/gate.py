"""The PR's pytest gate inside the pinned image (CPU), for hosts without torch.

    modal run -m miles_ext.disagg.gate --tests "miles_ext/tests tests/fast/utils/test_arguments.py"
"""

import os
import subprocess

import modal

from miles_ext.disagg.app import _ROOT, base_image

app = modal.App("miles-disagg-gate")
image = base_image.add_local_dir(
    _ROOT, "/root/wt", ignore=lambda p: p.parts[:1] in ((".git",), ("docs",)) or "__pycache__" in p.parts
)


@app.function(image=image, cpu=8, memory=16384, timeout=1800)
def gate(paths: list[str]) -> dict:
    env = {**os.environ, "PYTHONPATH": "/root/wt"}
    p = subprocess.run(
        ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", *paths],
        cwd="/root/wt",
        env=env,
        capture_output=True,
        text=True,
    )
    return {"returncode": p.returncode, "tail": (p.stdout + p.stderr)[-3000:]}


@app.local_entrypoint()
def main(tests: str):
    out = gate.remote(tests.split())
    print(out["tail"])
    print("returncode", out["returncode"])
