"""GLM-5.3-Flash engines draft with the release's NextN layer (EAGLE), except where they cannot."""

from pathlib import Path

import pytest

from miles_ext.recipes import recipe as R

GLM_STAK = Path(R.__file__).parent / "configs" / "glm53-flash-stak.yaml"
DRAFT_FLAGS = ("--sglang-speculative-algorithm", "--sglang-speculative-num-draft-tokens")


def _argv(monkeypatch, **overrides) -> list[str]:
    monkeypatch.setenv("MILES_DATA", "/data")
    return R.miles_args(R.load(GLM_STAK, overrides), "run1", resume=False)


@pytest.mark.parametrize(
    "overrides",
    [
        {"rollout": {"speculative": False}},
        {"model": {"preset": "glm-5.3-flash-4layer", "revision": None}},
    ],
    ids=["speculative-off", "no-nextn-layer"],
)
def test_the_fp8_release_drafts_except_where_excluded(monkeypatch, overrides):
    drafting = _argv(monkeypatch)
    assert drafting[drafting.index("--sglang-speculative-algorithm") + 1] == "EAGLE"
    assert drafting[drafting.index("--sglang-speculative-num-draft-tokens") + 1] == "4"
    excluded = _argv(monkeypatch, **overrides)
    assert not any(flag in excluded for flag in DRAFT_FLAGS)


def test_gpu_delta_drafts_only_when_the_recipe_opts_in(monkeypatch):
    split = {
        "cluster": {"placement": "split", "nodes": 1},
        "async": {"pause_generation": "retract"},
    }
    default = _argv(monkeypatch, **split, rollout={"weight_sync": "gpu-delta"})
    assert "--sglang-speculative-algorithm" not in default
    opted_in = _argv(monkeypatch, **split, rollout={"weight_sync": "gpu-delta", "speculative": True})
    assert opted_in[opted_in.index("--sglang-speculative-algorithm") + 1] == "EAGLE"
    assert "--sglang-enable-dp-attention" in opted_in


def test_rollout_speculative_is_null_or_a_bool():
    def errors(value):
        recipe = R.from_dict({"model": {"preset": "glm-5.3-flash"}, "rollout": {"speculative": value}})
        return [e for e in R.field_errors(recipe) if e["path"] == "rollout.speculative"]

    assert not any(errors(value) for value in (None, True, False))
    assert all(errors(value) for value in ("false", 0, [True]))
