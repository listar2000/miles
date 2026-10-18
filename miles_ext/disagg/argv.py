"""Miles argv for each deployment of a split run (primary, inference, trainer) and for the one-app baseline."""

import json
import os
import shlex
from dataclasses import asdict, dataclass, field

from miles.utils.object_store_config import compute_mooncake_init_kwargs_vanilla

PRIMARY, INFERENCE, TRAINER, ALL = "primary", "inference", "trainer", "all"
TRAINER_ID = "actor"
MODELS = "/models"
DATA = f"{MODELS}/datasets/dapo-math-17k/dapo-math-17k.jsonl"


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    run_uuid: str
    model: str = "Qwen3-0.6B"
    num_rollout: int = 3
    rollout_batch_size: int = 8
    n_samples: int = 4
    max_response_len: int = 1024
    train_gpus: int = 1
    engine_gpus: int = 1
    gpus_per_engine: int = 1
    fully_async: bool = True
    max_weight_staleness: int = 2
    seed: int = 1234
    save_dir: str = ""
    save_interval: int = 0
    wandb_project: str = "miles-disagg"
    expected_batch_s: int = 0  # conductor: rollout 0's expected duration, to start the trainer just in time
    extra: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def of(cls, d: dict) -> "RunConfig":
        return cls(**d)


def bench(run_id: str, run_uuid: str, **over) -> RunConfig:
    """The fixed benchmark every mode runs: Qwen3-4B math RL, 32 prompts x 8, 8k replies, 10 steps."""
    kw = dict(
        model="Qwen3-4B",
        num_rollout=10,
        rollout_batch_size=32,
        n_samples=8,
        max_response_len=8192,
        train_gpus=2,
        engine_gpus=2,
        max_weight_staleness=2,
    )
    return RunConfig(run_id=run_id, run_uuid=run_uuid, **{**kw, **over})


REPLAY_DIR = "/state/replay"


def replay(run_id: str, run_uuid: str, mode: str, **over) -> RunConfig:
    """Loss/grad-norm comparison of two paths on identical data: ``record`` saves 5 rollouts to the state volume,
    ``replay`` trains on them (fresh generation still runs but is discarded). ``<mode>:<tag>`` keeps a recording in
    its own folder."""
    mode, _, tag = mode.partition(":")
    data = f"{REPLAY_DIR}{'/' + tag if tag else ''}/rollout_{{rollout_id}}.pt"
    extra = {
        "record": ["--save-debug-rollout-data", data],
        "replay": [
            "--ci-inject-rollout-data-path",
            data,
            "--ci-inject-rollout-data-start-rollout-id",
            "0",
            # temperature-1 sampling: fresh tokens are not expected to match the recording
            "--ci-inject-rollout-data-min-match-ratio",
            "0",
        ],
    }[mode]
    kw = dict(
        model="Qwen3-4B",
        num_rollout=5,
        rollout_batch_size=8,
        n_samples=8,
        # Qwen3-4B needs ~8k tokens to reach an answer
        max_response_len=8192,
        train_gpus=2,
        engine_gpus=2,
        fully_async=False,
        extra=extra,
    )
    return RunConfig(run_id=run_id, run_uuid=run_uuid, **{**kw, **over})


def _shared(cfg: RunConfig) -> list[str]:
    a = [
        "--train-backend",
        "fsdp",
        "--hf-checkpoint",
        f"{MODELS}/{cfg.model}",
        "--prompt-data",
        DATA,
        "--input-key",
        "prompt",
        "--label-key",
        "label",
        "--apply-chat-template",
        "--rollout-shuffle",
        "--rm-type",
        "deepscaler",
        "--num-rollout",
        str(cfg.num_rollout),
        "--rollout-batch-size",
        str(cfg.rollout_batch_size),
        "--n-samples-per-prompt",
        str(cfg.n_samples),
        "--global-batch-size",
        str(cfg.rollout_batch_size * cfg.n_samples),
        "--rollout-max-response-len",
        str(cfg.max_response_len),
        "--rollout-temperature",
        "1",
        "--rollout-seed",
        str(cfg.seed),
        "--seed",
        str(cfg.seed),
        "--advantage-estimator",
        "grpo",
        "--eps-clip",
        "0.2",
        "--eps-clip-high",
        "0.28",
        "--optimizer",
        "adam",
        "--lr",
        "1e-6",
        "--lr-decay-style",
        "constant",
        "--weight-decay",
        "0.1",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.98",
        "--rollout-num-gpus-per-engine",
        str(cfg.gpus_per_engine),
        "--sglang-mem-fraction-static",
        "0.7",
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        str(cfg.train_gpus),
        "--gradient-checkpointing",
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu",
        "9216",
    ]
    if cfg.fully_async:
        a += ["--fully-async", "--max-weight-staleness", str(cfg.max_weight_staleness)]
    if cfg.save_dir:
        a += ["--save", cfg.save_dir, "--load", cfg.save_dir]
        if cfg.save_interval:
            a += ["--save-interval", str(cfg.save_interval)]
    if cfg.wandb_project:
        a += [
            "--use-wandb",
            "--wandb-team",
            "mobiuslab",
            "--wandb-project",
            cfg.wandb_project,
            "--wandb-group",
            cfg.run_id,
        ]
    return a + list(cfg.extra)


def _split(cfg: RunConfig, mooncake_master: str, *, segment: bool) -> list[str]:
    kwargs = compute_mooncake_init_kwargs_vanilla(
        host=mooncake_master.rsplit(":", 1)[0], master_port=int(mooncake_master.rsplit(":", 1)[1])
    )
    if not segment:
        kwargs["global_segment_size"] = "0"
    return [
        "--cluster-backend",
        "ray",
        "--worker-comm-backend",
        "rpc",
        "--run-uuid",
        cfg.run_uuid,
        "--object-store-backend",
        "mooncake",
        "--mooncake-store-init-kwargs",
        json.dumps(kwargs),
    ]


def build(
    cfg: RunConfig,
    component: str,
    *,
    mooncake_master: str = "",
    trainer_addr: str = "",
    inference_controller_addr: str = "",
    instance_id: str = "e0",
    trainer_segment: bool = True,
) -> list[str]:
    """argv after the entrypoint for one deployment; addresses are ``host:port`` (``[v6]:port`` for IPv6).
    ``trainer_segment=False``: the trainer contributes no Mooncake memory (a segment host holds the objects)."""
    if component == ALL:
        return _shared(cfg) + ["--rollout-num-gpus", str(cfg.engine_gpus)]
    segment = trainer_segment or component != TRAINER
    a = _shared(cfg) + _split(cfg, mooncake_master, segment=segment) + ["--deploy-component", component]
    if component == TRAINER:
        return a + ["--rollout-num-gpus", str(cfg.engine_gpus)]
    if component == INFERENCE:  # one inference deployment holds every engine of the run
        return a + [
            "--rollout-num-gpus",
            str(cfg.engine_gpus),
            "--deploy-instance-id",
            instance_id,
            "--inference-controller-addr",
            inference_controller_addr,
            # a late trainer skips the initial push: tag samples "0" (SGLang's "default" counts as unversioned)
            "--sglang-weight-version",
            "0",
        ]
    if component == PRIMARY:
        return a + [
            "--rollout-num-gpus",
            str(cfg.engine_gpus),
            "--api-server-port",
            "0",
            "--trainer-controller-addrs",
            f"{TRAINER_ID}={trainer_addr}",
            "--init-expected-num-cells",
            str(cfg.engine_gpus // cfg.gpus_per_engine),
        ]
    raise ValueError(f"unknown component {component!r}")


def host_port(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def show(argv: list[str]) -> str:
    return shlex.join(argv)


def wandb_env(cfg: RunConfig, mode: str) -> dict:
    """W&B tags (wandb.init reads them from env; Miles passes none) and the run's group."""
    commit = os.environ.get("MILES_MOBIUS_COMMIT", "unknown")[:10]
    tags = [
        f"recipe:disagg-{mode}",
        f"model:{cfg.model}",
        "sandbox:none",
        "harness:math",
        f"miles-mobius:{os.environ.get('MILES_MOBIUS_VERSION', 'unknown')}",
        f"commit:{commit}",
    ]
    return {"WANDB_TAGS": ",".join(tags), "WANDB_RUN_GROUP": cfg.run_id}
