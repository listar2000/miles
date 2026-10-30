"""Rollout ids at or above VAL_BASE are validation passes (miles_ext.sft.val); kept import-light for the rollout side."""

VAL_BASE = 1_000_000
ACTOR_CLASS = "miles_ext.sft.val.SftTrainRayActor"


def is_val(rollout_id: int) -> bool:
    return rollout_id >= VAL_BASE
