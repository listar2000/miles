"""The DeepSeek V4.1 overfitting-run recipe (ext-012): the argv and env it resolves to, and the
layouts it refuses. The Qwen recipes' argv is unchanged by the new fields (defaults) except the fully
async trainer's allocator."""

import json
from pathlib import Path

import pytest
import yaml

from miles_ext.recipes import recipe as R

ROOT = Path(__file__).resolve().parents[2]
DSV41 = "dsv41-flash-tb4-overfit16.yaml"
RELEASED = "/models/deepseek-ai--DeepSeek-V4.1-Flash-1017/1bae0e1088993b9b82de08f59de5be5a34e42170"


def _recipe(name: str = DSV41, **over) -> dict:
    spec = yaml.safe_load((ROOT / "miles_ext/recipes/configs" / name).read_text())
    spec["taskset"] = {k: v for k, v in spec["taskset"].items() if k not in ("tasks_dir", "prompts")}
    return R.from_dict(spec, over)


def _value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_dsv41_argv():
    r = _recipe()
    R.validate(r, local_inputs=False)
    argv = R.miles_args(r, "run1", resume=False)
    # the engines serve the released FP8 + FP4 checkpoint, the trainer loads the BF16 cast
    assert _value(argv, "--hf-checkpoint") == RELEASED
    assert _value(argv, "--ref-load") == "/models/DeepSeek-V4.1-Flash-1017-bf16"
    assert argv.count("--ref-load") == 1
    assert _value(argv, "--custom-weight-quantizer-path").endswith("quantize_like_checkpoint")
    assert _value(argv, "--custom-placement-hook-path") == "miles_ext.obs.gpu_telemetry.start"
    assert "--mxfp4-qat-routed-experts" in argv and "--stream-optimizer-state-to-disk" in argv
    assert _value(argv, "--stream-optimizer-state-moment-dtype") == "bf16"
    assert "--grad-reduce-in-bf16" in argv and "--accumulate-allreduce-grads-in-fp32" not in argv
    for flag, value in (
        ("--tensor-model-parallel-size", "4"),
        ("--pipeline-model-parallel-size", "2"),
        ("--context-parallel-size", "2"),
        ("--expert-model-parallel-size", "8"),
        ("--rollout-num-gpus-per-engine", "8"),
        ("--sglang-ep-size", "8"),
        ("--actor-num-nodes", "2"),
        ("--rollout-num-gpus", "16"),
        ("--micro-batch-size", "1"),
        ("--tis-clip-low", "0.5"),
        ("--max-weight-staleness", "4"),
        ("--sglang-context-length", "262144"),
        ("--max-seq-len", "262144"),
        ("--tito-model", "deepseekv41"),
    ):
        assert _value(argv, flag) == value, flag
    for flag in (
        "--allgather-cp",
        "--use-rollout-routing-replay",
        "--sglang-allow-auto-truncate",
        "--moe-router-freeze-gate",
        "--no-save-optim",
        "--no-log-inference-engine-weight-checksums",
    ):
        assert flag in argv, flag
    # text-only engines, so R3 capture runs (vision_topk skips it)
    assert json.loads(_value(argv, "--sglang-json-model-override-args")) == {"vision_n_layers": 0}
    assert _value(argv, "--dynamic-sampling-filter-path").endswith("check_reward_nonzero_std")
    for flag in ("--use-dynamic-batch-size", "--attention-backend", "--wandb-key", "--no-load-optim"):
        assert flag not in argv, flag
    assert "--no-load-optim" in R.miles_args(r, "run1", resume=True)
    env = R.trainer_env(r)
    assert env["SGLANG_DSV4_FP4_EXPERTS"] == "1" and env["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert R.harbor_env(r, "run1")["HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER"] == "3"


def test_dsv41_bf16_rollout_trains_and_serves_one_copy():
    r = _recipe(rollout={"precision": "bf16", "expert_qat": False})
    R.validate(r, local_inputs=False)
    argv = R.miles_args(r, "run1", resume=False)
    assert _value(argv, "--ref-load") == _value(argv, "--hf-checkpoint") == RELEASED
    assert "--custom-weight-quantizer-path" not in argv
    assert R.trainer_env(r)["SGLANG_DSV4_FP4_EXPERTS"] == "0"


def test_qwen_recipes_keep_their_defaults():
    argv = R.miles_args(_recipe("qwen35-9b-swegym8-megatron-2node.yaml"), "run1", resume=False)
    assert _value(argv, "--rollout-num-gpus-per-engine") == "1" and _value(argv, "--tis-clip-low") == "0.0"
    assert _value(argv, "--attention-backend") == "flash" and "--use-dynamic-batch-size" in argv
    assert _value(argv, "--dynamic-sampling-filter-path").endswith("check_no_aborted")
    assert "--accumulate-allreduce-grads-in-fp32" in argv
    for flag in (
        "--grad-reduce-in-bf16",
        "--expert-model-parallel-size",
        "--stream-optimizer-state-to-disk",
        "--no-save-optim",
        "--use-rollout-routing-replay",
        "--sglang-context-length",
        "--micro-batch-size",
    ):
        assert flag not in argv, flag
    assert R.trainer_env(_recipe("qwen35-9b-swegym8-sail.yaml")) == {"WANDB__DISABLE_STATS": "true"}


@pytest.mark.parametrize(
    "over, message",
    [
        ({"optimizer": {"offload": None}}, "optimizer.offload disk"),
        ({"parallel": {"ep": 4}}, "expert data parallel size 1"),
        ({"parallel": {"ep": 16}}, "ep 16 must divide"),
        ({"rollout": {"precision": "fp4"}}, "rollout.precision must be one of"),
        ({"rollout": {"gpus_per_engine": 3}}, "must divide 8"),
        ({"checkpoint": {"save_hf": True}}, "checkpoint.save_hf"),
        ({"cluster": {"disk_gib": None}}, "cluster.disk_gib"),
    ],
)
def test_dsv41_refuses(over, message):
    with pytest.raises(ValueError, match=message):
        R.validate(_recipe(**over), local_inputs=False)


def test_rollout_gpus_split_into_engines():
    with pytest.raises(ValueError, match="engines of 8"):  # 4 rollout GPUs on the one-node recipe
        R.validate(_recipe("qwen35-9b-swegym8-sail.yaml", rollout={"gpus_per_engine": 8}), local_inputs=False)


@pytest.mark.parametrize(
    "over, message",
    [
        (
            {"rollout": {"precision": "fp8_fp4"}},
            "rollout.precision must be one of",
        ),  # the Qwen presets serve bf16 only
        ({"optimizer": {"offload": "disk"}}, "megatron only"),
        ({"optimizer": {"grad_dtype": "bf16"}}, "megatron only"),
        ({"checkpoint": {"save_optimizer": False}}, "megatron only"),
    ],
)
def test_fsdp_refuses_megatron_only_fields(over, message):
    with pytest.raises(ValueError, match=message):
        R.validate(_recipe("qwen35-9b-swegym8-sail.yaml", **over), local_inputs=False)


def _task_ids(name: str) -> list[str]:
    lines = (ROOT / "miles_ext/recipes/configs" / name).read_text().splitlines()
    return [line.split()[0] for line in lines if line.strip() and not line.startswith("#")]


@pytest.mark.parametrize("name", ["dsv41-flash-tb4-overfit16", "dsv41-flash-overfit8", "dsv41-flash-overfit8-a3"])
def test_task_list_fills_a_step(name):
    ids = _task_ids(f"{name}.tasks")
    assert len(set(ids)) == len(ids) == _recipe(f"{name}.yaml")["algorithm"]["rollout_batch_size"]


@pytest.mark.parametrize("mode, expandable", [("fully_async", "True"), ("sync", "False")])
def test_trainer_allocator_follows_mode(mode, expandable):
    r = _recipe()
    r["algorithm"]["mode"] = mode  # torch_memory_saver (colocated) rejects expandable segments
    env = json.loads(_value(R.miles_args(r, "run1", resume=False), "--train-env-vars"))
    assert env == {"PYTORCH_CUDA_ALLOC_CONF": f"expandable_segments:{expandable}"}


def test_overfit8_one_step_per_rollout():
    r = _recipe("dsv41-flash-overfit8.yaml")
    R.validate(r, local_inputs=False)
    argv = R.miles_args(r, "run1", resume=False)
    # --skip-actor-forward-only needs exactly one optimizer step per rollout
    algo = r["algorithm"]
    assert "--skip-actor-forward-only" in argv
    assert int(_value(argv, "--global-batch-size")) == algo["rollout_batch_size"] * algo["n_samples"]
    assert _value(argv, "--save-debug-rollout-data") == "/results/run1/rollout_data/{rollout_id}.pt"
    r["checkpoint"]["save_rollout_data"] = False
    assert "--save-debug-rollout-data" not in R.miles_args(r, "run1", resume=False)


def test_overfit8_a3_score_centering():
    r = _recipe("dsv41-flash-overfit8-a3.yaml")
    R.validate(r, local_inputs=False)
    argv = R.miles_args(r, "run1", resume=False)
    # Miles rejects score centering with --use-tis or --skip-actor-forward-only
    assert _value(argv, "--loss-type") == "score_centering" and "--use-rollout-logprobs" in argv
    assert "--use-tis" not in argv and "--skip-actor-forward-only" not in argv
    assert "--disable-grpo-std-normalization" in argv and "--calculate-per-token-loss" in argv
    assert R.harbor_env(r, "run1")["MILES_EXT_CONTEXT_EXHAUSTED_SCORES_ZERO"] == "1"


def test_policy_loss_keeps_tis():
    r = _recipe("dsv41-flash-overfit8.yaml")
    argv = R.miles_args(r, "run1", resume=False)
    assert "--use-tis" in argv and "--loss-type" not in argv and "--rollout-top-logprobs-num" not in argv
    assert "--disable-grpo-std-normalization" not in argv and "--calculate-per-token-loss" not in argv
    assert "MILES_EXT_CONTEXT_EXHAUSTED_SCORES_ZERO" not in R.harbor_env(r, "run1")


def test_dsv41_lora_argv_passes_native_lora_startup(tmp_path, dsv41_lora_spec):
    from miles.backends.megatron_utils.lora.native import validate_native_lora_args

    lora = {"rank": 32, "alpha": 32}
    r = _recipe(lora=lora, optimizer={"offload": None, "moment_dtype": None}, rollout={"expert_qat": False})
    R.validate(r, local_inputs=False)
    argv = R.miles_args(r, "run1", resume=False)
    assert _value(argv, "--lora-rank") == "32" and "--lora-merged-sync" in argv
    assert _value(argv, "--custom-lora-spec-path") == "miles_ext.lora.dsv41"
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
    validate_native_lora_args(_native_lora_args(argv, hf_checkpoint=str(tmp_path)))
    with pytest.raises(ValueError, match="drop rollout.expert_qat"):
        R.validate(_recipe(lora=lora, optimizer={"offload": None}), local_inputs=False)


def _native_lora_args(argv: list[str], **overrides):
    """The flags native LoRA validates at startup, as Megatron parses them from ``argv``."""
    from argparse import Namespace

    flags = ("--colocate", "--overlap-param-gather", "--overlap-grad-reduce", "--moe-shared-expert-overlap")
    values = {flag.removeprefix("--").replace("-", "_"): flag in argv for flag in flags}
    return Namespace(
        **values,
        lora_rank=int(_value(argv, "--lora-rank")),
        lora_adapter_path=None,
        lora_train_only=False,
        lora_merged_sync="--lora-merged-sync" in argv,
        expert_tensor_parallel_size=int(_value(argv, "--expert-tensor-parallel-size")),
        **overrides,
    )


@pytest.mark.parametrize(
    "mode, buffer",
    [("log", "miles_ext.rollout.group_log.GroupLogDataBuffer"), ("prompts", None), ("groups", None)],
)
def test_checkpoint_buffer_modes(mode, buffer):
    r = _recipe(**{"async": {"checkpoint_buffer": mode}})
    R.validate(r, local_inputs=False)
    argv = R.miles_args(r, "run1", resume=False)
    assert _value(argv, "--async-checkpoint-buffer") == mode
    path = "--custom-async-data-buffer-path"
    assert (_value(argv, path) if path in argv else None) == buffer
