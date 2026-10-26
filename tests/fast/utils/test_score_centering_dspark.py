"""Score centering accepts DSpark speculative decoding and still refuses other speculative modes. [fork fix-058]"""

import pytest
from tests.fast.fixtures.score_centering_fixtures import _args

from miles.utils.score_centering import validate_score_centering_args


@pytest.mark.parametrize("algorithm", ["DSPARK", "dspark"])
def test_only_dspark_is_allowed(algorithm: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SGLANG_SIMULATE_ACC_LEN", raising=False)
    validate_score_centering_args(_args(sglang_speculative_algorithm=algorithm))
    for other in ("EAGLE", "EAGLE3", "NGRAM", "DFLASH"):
        with pytest.raises(ValueError, match="sglang-speculative-algorithm"):
            validate_score_centering_args(_args(sglang_speculative_algorithm=other))


@pytest.mark.parametrize("value,refused", [("3.5", True), ("1", True), ("-1", False), ("", False)])
def test_simulated_acceptance_is_refused(value: str, refused: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SGLANG_SIMULATE_ACC_LEN", value)
    args = _args(sglang_speculative_algorithm="DSPARK")
    if refused:
        with pytest.raises(ValueError, match="SGLANG_SIMULATE_ACC_LEN"):
            validate_score_centering_args(args)
    else:
        validate_score_centering_args(args)
