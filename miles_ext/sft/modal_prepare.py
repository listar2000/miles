"""The run config's records to token shards on a Modal CPU container. Usage: ``miles_ext/sft/README.md``."""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

from miles_ext import stack
from miles_ext.sft import config as sft_config
from miles_ext.sft import modal_common as M

app = modal.App("miles-rl-sft-prepare")
image = M.image()
weights_volume = M.volume("MILES_WEIGHTS_VOLUME")
data_volume = M.data_volume()
TESTS = (
    "miles_ext/tests/test_sft_record.py",
    "miles_ext/tests/test_sft_render.py",
    "miles_ext/tests/test_sft_prepare.py",
    "miles_ext/tests/test_sft_config.py",
)


def splits(records: Path) -> dict[str, list[str]]:
    files = sorted(str(p) for p in records.glob("*.jsonl") if not p.name.startswith("_"))
    return {
        "train": [f for f in files if not f.endswith(".val.jsonl")],
        "val": [f for f in files if f.endswith(".val.jsonl")],
    }


@app.function(
    image=image,
    cpu=16,
    memory=64 * 1024,
    timeout=3 * 3600,
    volumes={str(sft_config.WEIGHTS_MOUNT): weights_volume.read_only(), str(sft_config.DATA_MOUNT): data_volume},
)
def prepare(config_yaml: str, tests: bool) -> dict[str, str]:
    path = Path("/tmp/sft-config.yaml")
    path.write_text(config_yaml)
    cfg = sft_config.load(path)
    if tests:
        subprocess.run(["python", "-m", "pytest", "-q", "-x", *TESTS], cwd=str(stack.MILES_SRC), check=True)
    manifests = {}
    for split, inputs in splits(Path(sft_config.DATA_MOUNT / cfg.data.records)).items():
        if not inputs:
            continue
        out = Path(cfg.shards) / split
        argv = [
            "python",
            "-m",
            "miles_ext.sft.prepare",
            "--output",
            str(out),
            "--hf-checkpoint",
            str(cfg.weights),
            "--tito-model",
            cfg.preset.tito_model,
            "--max-tokens",
            str(cfg.data.max_tokens),
        ]
        for f in inputs:
            argv += ["--input", f]
        subprocess.run(argv, cwd=str(stack.MILES_SRC), check=True)
        manifests[split] = (out / "manifest.json").read_text()
    data_volume.commit()
    return manifests


@app.local_entrypoint()
def main(tests: bool = False):
    for split, manifest in prepare.remote(Path(M.CONFIG_PATH).read_text(), tests).items():
        print(split, manifest)
