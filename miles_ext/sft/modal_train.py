"""SFT of Qwen3-30B-A3B-Thinking-2507 on one Modal H200:8 node (Megatron, no inference engines).

    modal run --detach -m miles_ext.sft.modal_train --run-id <id> [--num-rollout N] > <id>.log 2>&1

The image is the fork's main build with this branch's ``miles_ext`` overlaid (``common.overlay``),
so nothing needs pushing. Inputs: the base weights on the weights volume
(``prepare::weights``) and the dataset on the data volume (``prepare::build``). Outputs under
``/results/<run_id>``: Megatron checkpoints (weights only), an HF export at every save
(``hf/iter_NNNNNNN``, tokenizer and configs copied from the base so serving metadata matches),
and ``argv.json``. The same run id again resumes from the last checkpoint (fresh optimizer state).

Layout: TP1, CP2 (ring attention over two GPUs), EP8, DP4; full recompute; packed THD batches
of up to 32K tokens per GPU (64K per CP pair, the longest row); the fused log-prob kernel (the
torch path keeps an fp32 softmax per chunk and ran out of memory at 32K tokens). One optimizer
step per rollout of ``BATCH`` trajectories, cosine LR 1e-5 -> 1e-6 with 50 warmup steps (the
recipe's card); about 16 s per step, 1,000 steps in about 5 h.
"""

import json
import os
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path

import modal

from miles_ext.sft import common as C

app = modal.App("miles-rl-sft")
image = C.overlay(C.base_image().env({"PYTHONPATH": "/root/miles:/root/Megatron-LM"}))
weights_volume = modal.Volume.from_name(C.WEIGHTS_VOLUME, version=2)
data_volume = modal.Volume.from_name(C.DATA_VOLUME, version=2)
results_volume = modal.Volume.from_name(C.RESULTS_VOLUME, create_if_missing=True, version=2)
secrets = [modal.Secret.from_dotenv(os.environ.get("MILES_DOTENV_DIR") or os.getcwd())]

BASE = Path(C.WEIGHTS_MOUNT) / C.BASE_DIR
DATA = Path(C.DATA_MOUNT) / C.DATASET_DIR
BATCH = 12  # ~ the recipe's 10.9 trajectories per step (4 x 128K packed blocks); divisible by DP 4
HF_META = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
           "merges.txt"]


def miles_args(run_id: str, num_rollout: int, save_interval: int, resume: bool, wandb_project: str) -> list[str]:
    from miles.utils.external_utils.model_args_utils import load_model_args

    run = Path(C.RESULTS_MOUNT) / run_id
    ckpt = run / "checkpoints"
    return shlex.split(f"""
        --train-backend megatron --hf-checkpoint {BASE} --ref-load {BASE}
        {load_model_args("qwen3-30B-A3B", rotary_base="10000000")}
        --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 2
        --expert-model-parallel-size 8 --expert-tensor-parallel-size 1
        --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
        --attention-backend flash --attention-softmax-in-fp32 --accumulate-allreduce-grads-in-fp32
        --attention-dropout 0.0 --hidden-dropout 0.0 --bf16
        --log-probs-backend fused
        --use-dynamic-batch-size --max-tokens-per-gpu 32768

        --debug-train-only --actor-num-nodes 1 --actor-num-gpus-per-node 8
        --rollout-function-path miles_ext.sft.rollout.generate_rollout
        --prompt-data {DATA / "train_prompts.jsonl"} --input-key text
        --rollout-shuffle --rollout-seed 0 --seed 0
        --num-rollout {num_rollout} --rollout-batch-size {BATCH} --n-samples-per-prompt 1 --global-batch-size {BATCH}
        --loss-type sft_loss --calculate-per-token-loss --disable-compute-advantages-and-returns

        --optimizer adam --lr 1e-5 --min-lr 1e-6 --lr-decay-style cosine --lr-warmup-iters {min(50, num_rollout // 2)}
        --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.999 --clip-grad 1.0
        --override-opt_param-scheduler

        --save {ckpt} --save-interval {save_interval} --no-save-optim
        {f"--load {ckpt} --no-load-optim" if resume else ""}
        --save-hf {run / "hf" / "iter_{rollout_id:07d}"}

        --use-wandb --wandb-project {wandb_project} --wandb-group {run_id}
    """)


def _committer(stop: threading.Event, every: int = 600):
    while not stop.wait(every):
        try:
            results_volume.commit()
        except Exception as e:  # a commit racing a checkpoint write retries on the next tick
            print(f"results commit: {type(e).__name__}: {e}", flush=True)


def _finish_hf_exports(run: Path):
    """Serving metadata of every HF export: the base's tokenizer and configs, byte for byte."""
    for d in sorted((run / "hf").glob("iter_*")):
        for name in HF_META:
            if (BASE / name).exists():
                shutil.copyfile(BASE / name, d / name)
        (d / "base_revision").write_text(C.BASE_REVISION)


@app.function(image=image, gpu="H200:8", cpu=64, memory=512 * 1024, secrets=secrets, timeout=24 * 3600,
              volumes={C.WEIGHTS_MOUNT: weights_volume.read_only(), C.DATA_MOUNT: data_volume.read_only(),
                       C.RESULTS_MOUNT: results_volume}, single_use_containers=True)
def train(run_id: str, num_rollout: int, save_interval: int, val_interval: int, val_rows: int,
          wandb_project: str):
    from miles_ext import stack

    run = Path(C.RESULTS_MOUNT) / run_id
    run.mkdir(parents=True, exist_ok=True)
    resume = (run / "checkpoints" / "latest_checkpointed_iteration.txt").exists()
    args = miles_args(run_id, num_rollout, save_interval, resume, wandb_project)
    (run / f"argv-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(
        {"image": C.FORK_IMAGE, "resume": resume, "argv": args}, indent=1))
    env = {**os.environ, "CUDA_DEVICE_MAX_CONNECTIONS": "1", "WANDB__DISABLE_STATS": "true",
           "MILES_SFT_DATA": str(DATA), "MILES_SFT_VAL_INTERVAL": str(val_interval),
           "MILES_SFT_VAL_ROWS": str(val_rows), "WANDB_RUN_GROUP": run_id, "WANDB_NAME": run_id,
           "NCCL_NVLS_ENABLE": "0", "PYTHONPATH": "/root/miles:/root/Megatron-LM"}
    subprocess.run(["ray", "start", "--head", "--num-gpus=8", "--disable-usage-stats"], env=env, check=True)
    stop = threading.Event()
    threading.Thread(target=_committer, args=(stop,), daemon=True).start()
    try:
        print("argv: " + " ".join(args), flush=True)
        rc = subprocess.run(["python", "-m", "miles_ext.sft.train", *args], cwd=str(stack.MILES_SRC), env=env).returncode
        _finish_hf_exports(run)
        if rc:
            raise RuntimeError(f"SFT exited with {rc}")
    finally:
        stop.set()
        results_volume.commit()


@app.local_entrypoint()
def main(run_id: str, num_rollout: int = 1000, save_interval: int = 250, val_interval: int = 100,
         val_rows: int = 252, wandb_project: str = "miles-sft"):
    call = train.spawn(run_id, num_rollout, save_interval, val_interval, val_rows, wandb_project)
    print(f"train call {call.object_id}", flush=True)
    call.get()
