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


DFLASH2 = {"model": {"preset": "glm-5.3-flash-dflash2"}}
BOUNDED = {"algorithm": {"top_p": 0.95, "top_k": 32}}


def test_dflash2_serves_its_own_checkpoint_with_top_p_replay(monkeypatch):
    argv = _argv(monkeypatch, **DFLASH2, **BOUNDED)
    assert argv[argv.index("--sglang-speculative-algorithm") + 1] == "DFLASH"
    draft = argv[argv.index("--sglang-speculative-draft-model-path") + 1]
    assert draft == "/models/incoai--GLM-5.3-Flash-DFlash2/bf582e4eacc1810f76656d1811693ff6c6737d2a"
    assert argv[argv.index("--rollout-top-p") + 1] == "0.95"
    assert argv[argv.index("--rollout-top-k") + 1] == "32"


@pytest.mark.parametrize(
    "overrides, message",
    [
        (BOUNDED, "exact-sampling draft"),
        ({"algorithm": {"top_k": 32}}, "exact-sampling draft"),
        ({"extra_args": ["--rollout-top-p", "0.95", "--rollout-top-k", "32"]}, "exact-sampling draft"),
        ({"extra_args": ["--rollout-top-p=0.95", "--rollout-top-k=32"]}, "exact-sampling draft"),
        ({"algorithm": {"loss": "score_centering"}}, "exact-sampling draft"),
        ({**DFLASH2, "algorithm": {"top_p": 0.95}}, "needs algorithm.top_k"),
    ],
    ids=[
        "eagle-top-p",
        "eagle-top-k",
        "eagle-extra-args",
        "eagle-extra-args-equals",
        "eagle-score-centering",
        "no-top-k",
    ],
)
def test_refused_sampling_combinations_fail_at_load(monkeypatch, overrides, message):
    with pytest.raises(ValueError, match=message):
        _argv(monkeypatch, **overrides)


def test_plain_decode_allows_every_sampling_option(monkeypatch):
    argv = _argv(
        monkeypatch, rollout={"speculative": False}, algorithm={"top_p": 0.95, "top_k": 32, "loss": "score_centering"}
    )
    assert "--sglang-speculative-algorithm" not in argv
    assert argv[argv.index("--loss-type") + 1] == "score_centering"


def test_dflash2_takes_score_centering_with_top_p(monkeypatch):
    argv = _argv(monkeypatch, **DFLASH2, algorithm={"top_p": 0.95, "top_k": 32, "loss": "score_centering"})
    assert argv[argv.index("--loss-type") + 1] == "score_centering"
    assert argv[argv.index("--sglang-speculative-algorithm") + 1] == "DFLASH"
