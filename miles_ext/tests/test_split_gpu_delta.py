"""ext-053: full-parameter split placement with GPU-delta weight sync (rollout.weight_sync gpu-delta)."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from miles_ext.launch.split import volume_sync
from miles_ext.recipes import recipe as R

ROOT = Path(__file__).resolve().parents[2]
SPLIT = "dsv41-flash-lora-split.yaml"


def _recipe(**over) -> dict:
    """``over`` uses dotted keys ("lora.rank"), nested into a recipe override."""
    spec = yaml.safe_load((ROOT / "miles_ext/recipes/configs" / SPLIT).read_text())
    spec["taskset"] = {k: v for k, v in spec["taskset"].items() if k not in ("tasks_dir", "prompts")}
    nested: dict = {}
    for key, value in over.items():
        section, field = key.split(".")
        nested.setdefault(section, {})[field] = value
    return R.from_dict(spec, nested)


def _full_param(**over) -> dict:
    return _recipe(
        **{
            "lora.rank": 0,
            "rollout.weight_sync": "gpu-delta",
            "async.pause_generation": "retract",
            "optimizer.offload": "disk",
            "cluster.disk_gib": 3072,
            **over,
        }
    )


@pytest.fixture
def mapped(monkeypatch):
    # the split LoRA recipe's preset stands in for a preset with an SGLang GPU-delta mapping
    name = _recipe()["model"]["preset"]
    monkeypatch.setattr(R, "GPU_DELTA_PRESETS", (name,))
    monkeypatch.setitem(R.PRESETS, name, replace(R.PRESETS[name], served_gib=476))


def test_gpu_delta_argv_and_env(mapped):
    r = _full_param()
    R.validate(r, local_inputs=False)
    argv = R.miles_args(r, "run1", resume=False)
    assert argv[argv.index("--update-weight-transfer-mode") + 1] == "gpu-delta"
    assert argv[argv.index("--update-weight-disk-dir") + 1] == "/results/run1/gpu_delta"
    assert argv[argv.index("--custom-update-weight-post-write-path") + 1] == R.GPU_DELTA_POST_WRITE
    assert "--custom-weight-transfer-protocol-path" not in argv
    assert argv[argv.index("--pause-generation-mode") + 1] == "retract"
    assert "--sglang-enable-dp-attention" in argv
    last = len(argv) - 1 - argv[::-1].index("--sglang-dp-size")  # argparse keeps the last value
    assert argv[last + 1] == str(r["rollout"]["gpus_per_engine"])
    assert R.trainer_env(r)["SGLANG_ENABLE_JIT_DEEPGEMM"] == "0"


@pytest.mark.parametrize(
    "over,error",
    [
        ({"lora.rank": 32}, "full-parameter only"),
        ({"async.pause_generation": "in_place"}, "retract"),
        (
            {"cluster.placement": "one_cluster", "cluster.nodes": 2, "cluster.train_gpus": 8},
            "needs cluster.placement split",
        ),
    ],
)
def test_gpu_delta_refusals(mapped, over, error):
    with pytest.raises(ValueError, match=error):
        R.validate(_full_param(**over), local_inputs=False)


def test_gpu_delta_needs_a_mapped_preset():
    with pytest.raises(ValueError, match="maps only"):
        R.validate(_full_param(), local_inputs=False)


def test_post_write_hook_commits_then_reloads_on_rank_0(monkeypatch):
    calls = []
    monkeypatch.setenv(volume_sync.AGENT_ENV, "http://127.0.0.1:9")
    monkeypatch.setattr(volume_sync, "request_commit", lambda name: calls.append(("commit", name)))
    monkeypatch.setattr(volume_sync, "_call", lambda url, body, timeout: calls.append(("reload", url, body)) or {})
    volume_sync.gpu_delta_post_write(None, "/results/run1/gpu_delta/stream/weight_v000002", [])
    assert calls == [
        ("commit", "weight_v000002"),
        ("reload", "http://127.0.0.1:9/reload", {"dir": "/results/run1/gpu_delta/stream/weight_v000002"}),
    ]
