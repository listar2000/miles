"""The SFT jobs a node runs (RFC 0005), with no provider in them: records to token shards, and training on them."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

from miles_ext import stack
from miles_ext.sft import config as sft_config

TESTS = (
    "miles_ext/tests/test_sft_record.py",
    "miles_ext/tests/test_sft_render.py",
    "miles_ext/tests/test_sft_prepare.py",
    "miles_ext/tests/test_sft_config.py",
)

# serving metadata copied from the base into every HF export (the Megatron export writes weights and config only)
HF_META = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "special_tokens_map.json",
)


def _load(config_yaml: str):
    path = Path("/tmp/sft-config.yaml")
    path.write_text(config_yaml)
    return sft_config.load(path)


def splits(records: Path) -> dict[str, list[str]]:
    files = sorted(str(p) for p in records.glob("*.jsonl") if not p.name.startswith("_"))
    return {
        "train": [f for f in files if not f.endswith(".val.jsonl")],
        "val": [f for f in files if f.endswith(".val.jsonl")],
    }


def prepare(config_yaml: str, tests: bool, commit: Callable[[], None]) -> dict[str, str]:
    """Write the config's records as token shards on the data mount; returns each split's manifest."""
    cfg = _load(config_yaml)
    if tests:
        subprocess.run(["python", "-m", "pytest", "-q", "-x", *TESTS], cwd=str(stack.MILES_SRC), check=True)
    manifests = {}
    for split, inputs in splits(Path(sft_config.DATA_MOUNT / cfg.data.records)).items():
        if not inputs:  # training validates before its first step, so both splits are required
            raise ValueError(f"no {split} records (*.jsonl / *.val.jsonl) under {cfg.data.records}")
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
        for proc in cfg.data.postprocess:
            argv += ["--postprocess", proc]
        subprocess.run(argv, cwd=str(stack.MILES_SRC), check=True)
        manifests[split] = (out / "manifest.json").read_text()
    commit()
    return manifests


def _committer(stop: threading.Event, run: Path, weights: Path, commit: Callable[[], None], every: int = 600) -> None:
    while not stop.wait(every):
        try:
            _finish_hf_exports(run, weights)  # each save's export becomes servable on the next tick
            commit()
        except Exception as e:  # a commit racing a checkpoint write retries on the next tick
            print(f"results commit: {type(e).__name__}: {e}", flush=True)


def _finish_hf_exports(run: Path, weights: Path) -> None:
    for d in sorted((run / "hf").glob("iter_*")):
        if not (d / "model.safetensors.index.json").exists() or (d / ".serving_meta").exists():
            continue  # still being written, or finished
        for name in HF_META:
            if (weights / name).exists():
                shutil.copyfile(weights / name, d / name)
        (d / ".serving_meta").touch()  # only after every file is in place


def train(config_yaml: str, run_id: str, num_rollout: int, save_interval: int, commit: Callable[[], None]) -> None:
    """One training job on this node, resuming if the run has a save."""
    cfg = _load(config_yaml)
    if "--custom-trainer-actor-class" not in (stack.MILES_SRC / "miles/utils/fork_arguments.py").read_text():
        raise RuntimeError(
            "the image predates hook-019 (--custom-trainer-actor-class): use a newer image or `commit:`"
        )
    run = Path(sft_config.RESULTS_MOUNT) / run_id
    run.mkdir(parents=True, exist_ok=True)
    resume = (run / "checkpoints" / "latest_checkpointed_iteration.txt").exists()
    argv = sft_config.miles_argv(cfg, run_id, num_rollout, save_interval, resume)
    (run / f"argv-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(
        json.dumps(
            {"image": cfg.image, "commit": cfg.commit, "config": config_yaml, "resume": resume, "argv": argv}, indent=1
        )
    )
    env = {
        **os.environ,
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "WANDB__DISABLE_STATS": "true",
        "MILES_SFT_VAL_DIR": str(cfg.shards / "val"),
        "MILES_SFT_VAL_INTERVAL": str(cfg.val.interval),
        "WANDB_RUN_GROUP": run_id,
        "WANDB_NAME": run_id,
        "NCCL_NVLS_ENABLE": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        **cfg.preset.env,
    }
    subprocess.run(
        ["ray", "start", "--head", f"--num-gpus={sft_config.GPUS}", "--disable-usage-stats"], env=env, check=True
    )
    stop = threading.Event()
    threading.Thread(target=_committer, args=(stop, run, Path(cfg.weights), commit), daemon=True).start()
    try:
        print("argv: " + " ".join(argv), flush=True)
        rc = subprocess.run(
            ["python", "-m", "miles_ext.sft.train", *argv], cwd=str(stack.MILES_SRC), env=env
        ).returncode
        _finish_hf_exports(run, Path(cfg.weights))
        if rc:
            raise RuntimeError(f"SFT exited with {rc}")
    finally:
        stop.set()
        commit()
