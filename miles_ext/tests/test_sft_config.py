import pytest
import yaml

from miles_ext.sft import config as sft_config

BASE = {
    "model": "qwen3-30b-a3b-thinking-2507",
    "image": "docker.io/x/y@sha256:" + "0" * 64,
    "data": {"volume": "v", "records": "sft/d/records", "shards": "sft/d/q-128k", "max_tokens": 131072},
    "train": {"cp": 4, "batch": 128},
    "tracking": {"wandb_project": "miles-sft"},
}


def _write(tmp_path, raw):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_tokens_per_gpu_follow_the_cap_and_cp(tmp_path):
    cfg = sft_config.load(_write(tmp_path, BASE))
    assert cfg.max_tokens_per_gpu == 32768
    assert str(cfg.weights) == "/models/Qwen3-30B-A3B-Thinking-2507"
    assert sft_config.schedule(cfg, samples=4326) == (66, 33)


@pytest.mark.parametrize(
    "change",
    [
        {"image": "docker.io/x/y:latest"},
        {"model": "no-such-model"},
        {"extra": 1},
        {"train": {"cp": 3}},
        {"commit": "abc"},
        {"image": None},
        {"model": "deepseek-v4.1-flash"},
        {"train": {"decay": "step"}},
    ],
)
def test_bad_configs_are_refused(tmp_path, change):
    with pytest.raises((ValueError, TypeError)):
        sft_config.load(_write(tmp_path, {**BASE, **change}))


def test_argv_carries_layout_schedule_and_preset_args(tmp_path):
    cfg = sft_config.load(_write(tmp_path, BASE))
    argv = sft_config.miles_argv(cfg, "r1", num_rollout=66, save_interval=33, resume=False)
    flag = lambda name: argv[len(argv) - 1 - argv[::-1].index(name) + 1]  # noqa: E731  the last one wins
    assert (flag("--context-parallel-size"), flag("--max-tokens-per-gpu"), flag("--rotary-base")) == (
        "4",
        "32768",
        "10000000",
    )
    assert (flag("--min-lr"), flag("--lr-decay-style"), flag("--lr-warmup-iters")) == ("0.0", "linear", "2")
    assert flag("--prompt-data") == "/datasets/sft/d/q-128k/train/prompts.jsonl"
    assert flag("--custom-trainer-actor-class") == "miles_ext.sft.val.SftTrainRayActor"
    assert "--load" not in argv
