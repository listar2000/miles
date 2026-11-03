"""Run check_gpu.py on one B200 in the miles-mobius image with this checkout's code overlaid.

    python -m modal run fork/probes/glm53_kpool/modal_check.py   # from the repo root; app name from MILES_PROBE_APP
"""

import os
import subprocess

import modal

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
IGNORE = ["**/__pycache__/**"]

image = modal.Image.from_registry("docker.io/starlimobius/miles-mobius:main")
for d in ("miles", "miles_plugins", "tests"):
    image = image.add_local_dir(f"{REPO}/{d}", f"/root/overlay/{d}", ignore=IGNORE)
image = image.add_local_file(os.path.join(os.path.dirname(__file__), "check_gpu.py"), "/root/check_gpu.py")
app = modal.App(os.environ.get("MILES_PROBE_APP", "miles-glm53-kpool-check"))


@app.function(image=image, gpu="B200", timeout=2400, cpu=8, memory=65536)
def run():
    subprocess.run(["python", "/root/check_gpu.py"], check=True, cwd="/root")


@app.local_entrypoint()
def main():
    run.remote()
