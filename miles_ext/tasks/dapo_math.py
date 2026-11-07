"""Single-turn grader for DAPO math: Miles' own check, in the shape ``miles_ext.tasks.single_turn`` reads."""

from miles.rollout.rm_hub.math_dapo_utils import compute_score


def grade(response: str, label: str) -> tuple[int, str]:
    """(reward, kind): kind is ok, wrong or no_answer."""
    result = compute_score(response, label)
    if result["acc"]:
        return 1, "ok"
    return 0, "no_answer" if result["pred"] == "[INVALID]" else "wrong"
