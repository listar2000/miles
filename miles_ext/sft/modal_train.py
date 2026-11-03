"""SFT on prepared shards, one Modal H200:8 node (Megatron, no inference engines), from an SFT run config.

    MILES_SFT_CONFIG=<config.yaml> MILES_WEIGHTS_VOLUME=... MILES_RESULTS_VOLUME=... \
        modal run --detach -m miles_ext.sft.modal_train --run-id <id> > <id>.log 2>&1

The config (``miles_ext.sft.config``) names the model preset, the image, the shards ``modal_prepare`` wrote and
the schedule. Outputs under ``/results/<run_id>``: Megatron checkpoints (weights only), an HF export at every
save (``hf/iter_NNNNNNN``, with the base's tokenizer and configs) and ``argv-*.json``. The same run id again
resumes from the last checkpoint with fresh optimizer state. With ``MILES_MONITOR_DIR`` set the run is
registered on the monitor as kind ``sft``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import modal

from miles_ext import stack
from miles_ext.sft import config as sft_config
from miles_ext.sft import modal_common as M

app = modal.App("miles-rl-sft")
image = M.image()
weights_volume = M.volume("MILES_WEIGHTS_VOLUME")
results_volume = M.volume("MILES_RESULTS_VOLUME")
data_volume = M.data_volume()

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


def _committer(stop: threading.Event, every: int = 600) -> None:
    while not stop.wait(every):
        try:
            results_volume.commit()
        except Exception as e:  # a commit racing a checkpoint write retries on the next tick
            print(f"results commit: {type(e).__name__}: {e}", flush=True)


def _finish_hf_exports(run: Path, weights: Path) -> None:
    for d in sorted((run / "hf").glob("iter_*")):
        for name in HF_META:
            if (weights / name).exists():
                shutil.copyfile(weights / name, d / name)


@app.function(
    image=image,
    gpu="H200:8",
    cpu=64,
    memory=512 * 1024,
    timeout=24 * 3600,
    secrets=[modal.Secret.from_dotenv(os.environ.get("MILES_DOTENV_DIR") or os.getcwd())],
    volumes={
        str(sft_config.WEIGHTS_MOUNT): weights_volume.read_only(),
        str(sft_config.DATA_MOUNT): data_volume.read_only(),
        str(sft_config.RESULTS_MOUNT): results_volume,
    },
    single_use_containers=True,
)
def train(config_yaml: str, run_id: str, num_rollout: int, save_interval: int) -> None:
    path = Path("/tmp/sft-config.yaml")
    path.write_text(config_yaml)
    cfg = sft_config.load(path)
    run = Path(sft_config.RESULTS_MOUNT) / run_id
    run.mkdir(parents=True, exist_ok=True)
    resume = (run / "checkpoints" / "latest_checkpointed_iteration.txt").exists()
    argv = sft_config.miles_argv(cfg, run_id, num_rollout, save_interval, resume)
    (run / f"argv-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(
        json.dumps({"image": cfg.image, "config": config_yaml, "resume": resume, "argv": argv}, indent=1)
    )
    env = {
        **os.environ,
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "WANDB__DISABLE_STATS": "true",
        "MILES_SFT_VAL_DIR": str(cfg.shards / "val"),
        "MILES_SFT_VAL_INTERVAL": str(cfg.val.interval),
        "MILES_SFT_VAL_ROWS": str(cfg.val.rows),
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
    threading.Thread(target=_committer, args=(stop,), daemon=True).start()
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
        results_volume.commit()


@app.local_entrypoint()
def main(run_id: str, num_rollout: int = 0):
    """``--num-rollout`` overrides the config's epochs (a smoke run: 2)."""
    cfg = M.CONFIG
    manifest = json.loads(b"".join(data_volume.read_file(f"/{cfg.data.shards}/train/manifest.json")))
    steps, save_interval = sft_config.schedule(cfg, manifest["samples"])
    if num_rollout:
        steps, save_interval = num_rollout, num_rollout
    print(f"{run_id}: {manifest['samples']} samples, {steps} steps, save every {save_interval}", flush=True)
    call = train.spawn(Path(M.CONFIG_PATH).read_text(), run_id, steps, save_interval)
    print(f"train call {call.object_id}", flush=True)
    if os.environ.get("MILES_MONITOR_DIR"):
        from miles_ext.monitor import manifest as monitor

        monitor.write_manifest(
            app.app_id,
            run_id,
            kind="sft",
            expected_steps=steps,
            gpu=f"1x H200:{sft_config.GPUS}",
            gpu_app_name=app.name,
            wandb_project=sft_config.wandb_project(cfg, run_id),
            hard_stop_h=24,
            title=f"SFT {cfg.model} on {cfg.data.shards} ({cfg.train.epochs:g} ep, bs {cfg.train.batch}, lr {cfg.train.lr:g})",
        )
    call.get()
