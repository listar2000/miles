"""The SFT run config (YAML) that ``modal_prepare`` and ``modal_train`` read, and the Miles argv it gives."""

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


@dataclass(frozen=True)
class Data:
    volume: str
    records: str
    shards: str
    max_tokens: int
    postprocess: tuple[str, ...] = ()  # record processors by dotted path (prepare --postprocess), in order


@dataclass(frozen=True)
class Train:
    epochs: float = 2.0
    batch: int = 128
    lr: float = 2e-5
    decay: str = "linear"
    warmup: float = 0.03
    cp: int = 2


@dataclass(frozen=True)
class Val:
    interval: int = 20


@dataclass(frozen=True)
class SftConfig:
    model: str
    image: str | None  # a published fork image, pinned by digest
    commit: str | None  # or a pushed fork commit, built into an image on Modal
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
        return math.ceil(self.data.max_tokens / self.train.cp)


def load(path: str | Path) -> SftConfig:
    raw = yaml.safe_load(Path(path).read_text())
    unknown = set(raw) - {"model", "image", "commit", "data", "train", "val", "tracking"}
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    cfg = SftConfig(
        model=raw["model"],
        image=raw.get("image"),
        commit=raw.get("commit"),
        data=Data(**{**raw["data"], "postprocess": tuple(raw["data"].get("postprocess", ()))}),
        wandb_project=raw["tracking"]["wandb_project"],
        train=Train(**raw.get("train", {})),
        val=Val(**raw.get("val", {})),
    )
    if cfg.model not in PRESETS or cfg.preset.megatron_model_type is None:
        raise ValueError(f"{path}: model {cfg.model!r} is not a preset with a Megatron definition")
    if (cfg.image is None) == (cfg.commit is None):
        raise ValueError(f"{path}: give one of image (repo@sha256:...) or commit")
    if cfg.image is not None and "@sha256:" not in cfg.image:
        raise ValueError(f"{path}: image must be pinned by digest (repo@sha256:...)")
    if not cfg.preset.packed_batches:
        raise ValueError(f"{path}: {cfg.model} trains one sample per micro-batch, which SFT does not support yet")
    if cfg.train.decay not in ("linear", "cosine"):
        raise ValueError(f"{path}: train.decay is linear or cosine")
    if GPUS % cfg.train.cp:
        raise ValueError(f"{path}: train.cp must divide {GPUS}")
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
        --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
        --attention-softmax-in-fp32 --accumulate-allreduce-grads-in-fp32
        --attention-dropout 0.0 --hidden-dropout 0.0 --bf16
        --log-probs-backend fused
        --use-dynamic-batch-size --max-tokens-per-gpu {cfg.max_tokens_per_gpu}

        --debug-train-only --actor-num-nodes 1 --actor-num-gpus-per-node {GPUS}
        --custom-trainer-actor-class miles_ext.sft.val.SftTrainRayActor
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
    return SMOKE_PROJECT if run_id.startswith(SMOKE_PREFIXES) else cfg.wandb_project


def schedule(cfg: SftConfig, samples: int) -> tuple[int, int]:
    """(num_rollout, save_interval) for *samples* prepared training samples: a save at the end of each epoch."""
    steps_per_epoch = max(1, samples // cfg.train.batch)
    return max(1, round(cfg.train.epochs * steps_per_epoch)), steps_per_epoch
