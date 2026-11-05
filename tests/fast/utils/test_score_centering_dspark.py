"""Validate score-centering acceptance and refusal of speculative decoding configurations. [fork fix-058] [fork fix-085]"""

import json
from pathlib import Path

import pytest
from tests.fast.fixtures.score_centering_fixtures import _args

from miles.utils.score_centering import validate_score_centering_args


def _dflash2_draft(path: Path, architecture: str = "DFlash2DraftModel") -> str:
    (path / "config.json").write_text(json.dumps({"architectures": [architecture]}))
    return str(path)


# [fork fix-085]
@pytest.mark.parametrize("algorithm", ["DSPARK", "dspark", "DFLASH"])
def test_only_exact_drafts_are_allowed(algorithm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SGLANG_SIMULATE_ACC_LEN", raising=False)
    validate_score_centering_args(
        _args(sglang_speculative_algorithm=algorithm, sglang_speculative_draft_model_path=_dflash2_draft(tmp_path))
    )
    for other in ("EAGLE", "EAGLE3", "NGRAM"):
        with pytest.raises(ValueError, match="sglang-speculative-algorithm"):
            validate_score_centering_args(_args(sglang_speculative_algorithm=other))


# [fork fix-085]
@pytest.mark.parametrize("field", ["single", "acc"])
def test_dflash_with_lowered_thresholds_is_refused(
    field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SGLANG_SIMULATE_ACC_LEN", raising=False)
    args = _args(
        sglang_speculative_algorithm="DFLASH",
        sglang_speculative_draft_model_path=_dflash2_draft(tmp_path),
        **{f"sglang_speculative_accept_threshold_{field}": 0.5},
    )
    with pytest.raises(ValueError, match="thresholds"):
        validate_score_centering_args(args)


# [fork fix-085]
def test_dflash_without_the_selector_draft_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SGLANG_SIMULATE_ACC_LEN", raising=False)
    for path in (_dflash2_draft(tmp_path, "DFlashDraftModel"), str(tmp_path / "missing")):
        args = _args(sglang_speculative_algorithm="DFLASH", sglang_speculative_draft_model_path=path)
        with pytest.raises(ValueError, match="DFlash2"):
            validate_score_centering_args(args)


@pytest.mark.parametrize("value,refused", [("3.5", True), ("1", True), ("-1", False), ("", False)])
def test_simulated_acceptance_is_refused(value: str, refused: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SGLANG_SIMULATE_ACC_LEN", value)
    args = _args(sglang_speculative_algorithm="DSPARK")
    if refused:
        with pytest.raises(ValueError, match="SGLANG_SIMULATE_ACC_LEN"):
            validate_score_centering_args(args)
    else:
        validate_score_centering_args(args)
