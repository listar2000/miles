"""An SFT run config (YAML): model preset, image, data on a volume, schedule. One file drives the whole run:
``modal_prepare`` (records -> shards) and ``modal_train`` (shards -> checkpoints) both read it.

    model: qwen3-30b-a3b-thinking-2507       # miles_ext.recipes.presets.PRESETS key
    image: docker.io/starlimobius/miles-mobius@sha256:...   # pinned; this checkout's miles_ext is overlaid
    data:
      volume: miles-rl-datasets
      records: sft/<dataset>/records         # *.jsonl train records, *.val.jsonl held out, _* ignored
      shards: sft/<dataset>/<model>-128k     # prepare writes train/ and val/ here
      max_tokens: 131072                     # longer samples are dropped whole (prepare --max-tokens)
    train:
      epochs: 2
      batch: 128                             # trajectories per optimizer step
      lr: 2.0e-5
      decay: linear                          # linear (to 0) or cosine (to lr/10)
      warmup: 0.03                           # fraction of the steps
      cp: 4                                  # context parallel; max_tokens / cp tokens per GPU
      saves_per_epoch: 1
    val: {interval: 20, rows: 0}             # held-out NLL every <interval> steps; rows 0: all
    tracking: {wandb_project: miles-sft}

The layout is one H200:8 node, TP1 CP<cp> EP8, full recompute, packed batches; weights come from
``MILES_WEIGHTS_VOLUME`` at ``/models/<preset dir>`` and results go to ``MILES_RESULTS_VOLUME``.
"""

from __future__ import annotations

import math
import shlex
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import yaml

from miles_ext.recipes.presets import PRESETS, Preset
from miles_ext.recipes.recipe import SMOKE_PREFIXES, SMOKE_PROJECT

WEIGHTS_MOUNT = PurePosixPath("/models")
DATA_MOUNT = PurePosixPath("/datasets")
RESULTS_MOUNT = PurePosixPath("/results")
GPUS = 8
RECOMPUTE = {
    "full": "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1",
    "selective": "--recompute-granularity selective --recompute-modules moe_act layernorm",
}


@dataclass(frozen=True)
class Data:
    volume: str
    records: str
    shards: str
    max_tokens: int


@dataclass(frozen=True)
class Train:
    epochs: float = 2.0
    batch: int = 128
    lr: float = 2e-5
    decay: str = "linear"
    warmup: float = 0.03
    cp: int = 2
    saves_per_epoch: int = 1
    # "full" (every layer), "selective" (MoE activations and norms only) or N (the first N layers of each stage, in full)
    recompute: str | int = "full"
    # packed tokens per GPU and micro-batch; at least data.max_tokens / cp (the default)
    tokens_per_gpu: int | None = None


@dataclass(frozen=True)
class Val:
    interval: int = 20
    rows: int = 0


@dataclass(frozen=True)
class SftConfig:
    model: str
    image: str
    data: Data
    wandb_project: str
    train: Train = field(default_factory=Train)
    val: Val = field(default_factory=Val)

    @property
    def preset(self) -> Preset:
        return PRESETS[self.model]

    @property
    def weights(self) -> PurePosixPath:
        return WEIGHTS_MOUNT / (self.preset.train_dir_name or self.preset.dir_name)

    @property
    def shards(self) -> PurePosixPath:
        return DATA_MOUNT / self.data.shards

    @property
    def max_tokens_per_gpu(self) -> int:
        return self.train.tokens_per_gpu or math.ceil(self.data.max_tokens / self.train.cp)


def load(path: str | Path) -> SftConfig:
    raw = yaml.safe_load(Path(path).read_text())
    unknown = set(raw) - {"model", "image", "data", "train", "val", "tracking"}
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    cfg = SftConfig(
        model=raw["model"],
        image=raw["image"],
        data=Data(**raw["data"]),
        wandb_project=raw["tracking"]["wandb_project"],
        train=Train(**raw.get("train", {})),
        val=Val(**raw.get("val", {})),
    )
    if cfg.model not in PRESETS or cfg.preset.megatron_model_type is None:
        raise ValueError(f"{path}: model {cfg.model!r} is not a preset with a Megatron definition")
    if "@sha256:" not in cfg.image:
        raise ValueError(f"{path}: image must be pinned by digest (repo@sha256:...)")
    if cfg.train.decay not in ("linear", "cosine"):
        raise ValueError(f"{path}: train.decay is linear or cosine")
    if GPUS % cfg.train.cp:
        raise ValueError(f"{path}: train.cp must divide {GPUS}")
    if cfg.train.tokens_per_gpu and cfg.train.tokens_per_gpu * cfg.train.cp < cfg.data.max_tokens:
        raise ValueError(f"{path}: train.tokens_per_gpu * train.cp must hold data.max_tokens")
    if not (cfg.train.recompute in ("full", "selective") or isinstance(cfg.train.recompute, int)):
        raise ValueError(f"{path}: train.recompute is full, selective or a layer count")
    return cfg


def miles_argv(cfg: SftConfig, run_id: str, num_rollout: int, save_interval: int, resume: bool) -> list[str]:
    """The Miles argv of ``miles_ext.sft.train`` for this run."""
    from miles.utils.external_utils.model_args_utils import load_model_args

    t, p = cfg.train, cfg.preset
    run = RESULTS_MOUNT / run_id
    ckpt = run / "checkpoints"
    min_lr = t.lr / 10 if t.decay == "cosine" else 0.0
    return shlex.split(
        f"""
        --train-backend megatron --hf-checkpoint {cfg.weights} --ref-load {cfg.weights}
        {load_model_args(p.megatron_model_type)} {p.megatron_args} {p.megatron_cp_args if t.cp > 1 else ""}
        --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size {t.cp}
        --expert-model-parallel-size {GPUS} --expert-tensor-parallel-size 1
        {f"--attention-backend {p.attention_backend}" if p.attention_backend else ""}
        {RECOMPUTE.get(t.recompute) or f"--recompute-granularity full --recompute-method block --recompute-num-layers {t.recompute}"}
        --attention-softmax-in-fp32 --accumulate-allreduce-grads-in-fp32
        --attention-dropout 0.0 --hidden-dropout 0.0 --bf16
        --log-probs-backend fused
        --use-dynamic-batch-size --max-tokens-per-gpu {cfg.max_tokens_per_gpu}

        --debug-train-only --actor-num-nodes 1 --actor-num-gpus-per-node {GPUS}
        --rollout-function-path miles_ext.sft.rollout.generate_rollout
        --prompt-data {cfg.shards / "train" / "prompts.jsonl"} --input-key text
        --rollout-shuffle --rollout-seed 0 --seed 0
        --num-rollout {num_rollout} --rollout-batch-size {t.batch} --n-samples-per-prompt 1 --global-batch-size {t.batch}
        --loss-type sft_loss --calculate-per-token-loss --disable-compute-advantages-and-returns

        --optimizer adam --lr {t.lr} --min-lr {min_lr} --lr-decay-style {t.decay}
        --lr-warmup-iters {max(1, round(num_rollout * t.warmup))}
        --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.999 --clip-grad 1.0
        --override-opt_param-scheduler

        --save {ckpt} --save-interval {save_interval} --no-save-optim
        {f"--load {ckpt} --no-load-optim" if resume else ""}
        --save-hf {run / "hf" / "iter_{rollout_id:07d}"}

        --use-wandb --wandb-project {wandb_project(cfg, run_id)} --wandb-group {run_id}
    """
    )


def wandb_project(cfg: SftConfig, run_id: str) -> str:
    """``smoke-``/``debug-`` run ids log to the smoke project, as RL launches do."""
    return SMOKE_PROJECT if run_id.startswith(SMOKE_PREFIXES) else cfg.wandb_project


def schedule(cfg: SftConfig, samples: int) -> tuple[int, int]:
    """(num_rollout, save_interval) for *samples* prepared training samples."""
    steps_per_epoch = max(1, samples // cfg.train.batch)
    return max(1, round(cfg.train.epochs * steps_per_epoch)), max(1, steps_per_epoch // cfg.train.saves_per_epoch)
