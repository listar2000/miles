"""Rollout function for SFT on pre-rendered trajectories (``--rollout-function-path``).

The prompt file has one row per training trajectory, ``{"text": "<row id>"}``; the data
source shuffles and serves them as usual. This function looks each id up in the dataset
built by ``miles_ext.sft.prepare`` and returns a Sample with the stored token ids and loss
mask, so nothing is re-rendered during training. Rollout ids at or above ``val.VAL_BASE``
return the validation rows instead (``miles_ext.sft.val``).

Env: ``MILES_SFT_DATA`` (the dataset dir), ``MILES_SFT_VAL_ROWS`` (validation rows used, the
first N by id; 0 = all).
"""

import json
import logging
import os
from functools import cache
from pathlib import Path

import numpy as np

from miles.utils.types import Sample
from miles_ext.sft.ids import is_val

logger = logging.getLogger(__name__)


@cache
def _index(split: str) -> dict[int, dict]:
    path = Path(os.environ["MILES_SFT_DATA"]) / f"{split}.jsonl"
    return {r["id"]: r for r in map(json.loads, open(path))}


@cache
def _shard(shard: int):
    d = Path(os.environ["MILES_SFT_DATA"]) / "shards"
    return np.load(d / f"{shard:05d}.tokens.npy", mmap_mode="r"), np.load(d / f"{shard:05d}.mask.npy", mmap_mode="r")


def _fill(sample: Sample, row: dict) -> Sample:
    tokens, masks = _shard(row["shard"])
    a, b = row["offset"], row["offset"] + row["length"]
    ids, mask = np.asarray(tokens[a:b]).tolist(), np.asarray(masks[a:b]).tolist()
    first = mask.index(1)
    sample.tokens = ids
    sample.response_length = len(ids) - first
    sample.loss_mask = mask[first:]
    sample.reward = 0
    return sample


def _val_samples() -> list[list[Sample]]:
    rows = sorted(_index("val").values(), key=lambda r: r["id"])
    n = int(os.environ.get("MILES_SFT_VAL_ROWS", "0"))
    rows = rows[:n] if n else rows
    return [[_fill(Sample(group_index=i, index=i, prompt=str(r["id"])), r)] for i, r in enumerate(rows)]


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    assert not evaluation
    if is_val(rollout_id):
        return _val_samples()
    index = _index("train")
    groups = data_buffer.get_samples(args.rollout_batch_size)
    for group in groups:
        (sample,) = group
        _fill(sample, index[int(sample.prompt)])
    if rollout_id == 0:
        s = groups[0][0]
        logger.info(f"sft rollout: {len(groups)} rows, first {s.prompt}: {len(s.tokens)} tokens, "
                    f"{sum(s.loss_mask)} trained")
    return groups
