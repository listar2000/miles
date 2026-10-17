"""The DeepSeek V4.1 overfitting-run recipe (ext-012): the argv and env it resolves to, and the
layouts it refuses. The Qwen recipes' argv is unchanged by the new fields (defaults)."""

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
    assert R.trainer_env(_recipe("qwen35-9b-swegym8-sail.yaml")) == {}


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


def test_overfit_task_list_fills_a_step():
    tasks = (ROOT / "miles_ext/recipes/configs/dsv41-flash-tb4-overfit16.tasks").read_text().splitlines()
    ids = [line.split()[0] for line in tasks if line.strip() and not line.startswith("#")]
    assert len(set(ids)) == len(ids) == _recipe()["algorithm"]["rollout_batch_size"]
