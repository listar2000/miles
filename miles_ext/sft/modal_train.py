"""SFT on the shards of the run config ``MILES_SFT_CONFIG`` names, on one Modal H200:8 node."""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

from miles_ext.sft import config as sft_config
from miles_ext.sft import job
from miles_ext.sft import modal_common as M

app = modal.App("miles-rl-sft")
image = M.image()
weights_volume = M.volume("MILES_WEIGHTS_VOLUME")
results_volume = M.volume("MILES_RESULTS_VOLUME")
data_volume = M.data_volume()


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
    job.train(config_yaml, run_id, num_rollout, save_interval, results_volume.commit)


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
