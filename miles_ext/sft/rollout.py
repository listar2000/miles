"""Serve prepared SFT samples from the shards beside ``--prompt-data`` to ``--debug-train-only`` training."""

from __future__ import annotations

import json
import logging
from functools import cache
from pathlib import Path

import numpy as np

from miles.utils.types import Sample

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)


@cache
def _index(directory: str) -> dict[int, dict]:
    with open(Path(directory) / "samples.jsonl") as f:
        return {row["id"]: row for row in map(json.loads, f)}


@cache
def _shard(directory: str, shard: int) -> tuple[np.ndarray, np.ndarray]:
    shards = Path(directory) / "shards"
    return (
        np.load(shards / f"{shard:05d}.tokens.npy", mmap_mode="r"),
        np.load(shards / f"{shard:05d}.mask.npy", mmap_mode="r"),
    )


def fill_sample(sample: Sample, directory: str) -> Sample:
    """Set *sample*'s tokens and loss mask from the prepared sample its prompt names."""
    row = _index(directory)[int(sample.prompt)]
    tokens, masks = _shard(directory, row["shard"])
    start, end = row["offset"], row["offset"] + row["length"]
    token_ids = np.asarray(tokens[start:end]).tolist()
    mask = np.asarray(masks[start:end]).tolist()
    response_start = mask.index(1)
    sample.tokens = token_ids
    sample.response_length = len(token_ids) - response_start
    sample.loss_mask = mask[response_start:]
    sample.reward = 0
    return sample


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    assert not evaluation, "SFT samples have no evaluation rollout"
    directory = str(Path(args.prompt_data).parent)
    groups = data_buffer.get_samples(args.rollout_batch_size)
    for group in groups:
        for sample in group:
            fill_sample(sample, directory)
    if rollout_id == 0 and groups:
        first = groups[0][0]
        logger.info(
            "sft rollout: %d samples from %s; first has %d tokens, %d trained",
            len(groups),
            directory,
            len(first.tokens),
            sum(first.loss_mask),
        )
    return groups
