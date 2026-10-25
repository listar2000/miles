"""The split recipe: its argv, its Modal shape, and what split placement refuses."""

from pathlib import Path

import pytest

from miles_ext.recipes import recipe as R

CONFIGS = Path(__file__).resolve().parents[1] / "recipes" / "configs"


def _split(monkeypatch, **overrides) -> dict:
    monkeypatch.setenv("MILES_DATA", "/data")
    return R.load(CONFIGS / "dsv41-flash-lora-split.yaml", overrides)


def test_split_argv_and_shape(monkeypatch):
    recipe = _split(monkeypatch)
    argv = R.miles_args(recipe, "run1", resume=False)
    i = argv.index("--rollout-external-engine-addrs")
    assert argv[i + 1 : i + 3] == ["127.0.0.1:31000", "127.0.0.1:31001"]
    assert argv[argv.index("--custom-weight-transfer-protocol-path") + 1] == R.SPLIT_SYNC_PROTOCOL
    assert argv[argv.index("--rollout-num-gpus") + 1] == "8"
    assert argv[argv.index("--actor-num-nodes") + 1] == "1"
    shape = R.shape(recipe)
    assert shape["nodes"] == 1 and shape["placement"] == R.SPLIT
    assert shape["engine"] == {"gpu": "B300", "cpu": 64, "memory_gib": 768, "disk_gib": None, "timeout_hours": 10}


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"lora": {"serving": "merged"}}, "lora.serving adapter only"),
        ({"lora": {"rank": 0}}, "lora.serving adapter only"),
        ({"cluster": {"nodes": 2}}, "one trainer node"),
        ({"cluster": {"train_gpus": 4}}, "whole node"),
    ],
)
def test_split_refuses_what_it_cannot_sync(monkeypatch, overrides, message):
    with pytest.raises(ValueError, match=message):
        _split(monkeypatch, **overrides)


def test_one_cluster_has_no_split_args(monkeypatch):
    recipe = _split(monkeypatch, cluster={"placement": "one_cluster", "nodes": 2})
    argv = R.miles_args(recipe, "run1", resume=False)
    assert "--rollout-external-engine-addrs" not in argv and "--custom-weight-transfer-protocol-path" not in argv
    assert "placement" not in R.shape(recipe)
