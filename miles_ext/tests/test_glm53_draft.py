"""GLM-5.3-Flash engines draft with the release's NextN layer (EAGLE), except where it has not been run."""

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
        {
            "cluster": {"placement": "split", "nodes": 1},
            "rollout": {"weight_sync": "gpu-delta"},
            "async": {"pause_generation": "retract"},
        },
    ],
    ids=["speculative-off", "no-nextn-layer", "gpu-delta"],
)
def test_the_fp8_release_drafts_except_where_excluded(monkeypatch, overrides):
    drafting = _argv(monkeypatch)
    assert drafting[drafting.index("--sglang-speculative-algorithm") + 1] == "EAGLE"
    assert drafting[drafting.index("--sglang-speculative-num-draft-tokens") + 1] == "4"
    excluded = _argv(monkeypatch, **overrides)
    assert not any(flag in excluded for flag in DRAFT_FLAGS)
