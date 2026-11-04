"""Serve prepared SFT samples from the shards beside ``--prompt-data`` to ``--debug-train-only`` training.

Rollout ids at or above ``VAL_BASE`` get the held-out samples of ``MILES_SFT_VAL_DIR`` (set by ``modal_train``)."""

from __future__ import annotations

import json
import logging
import os
from functools import cache
from pathlib import Path

import numpy as np

from miles.utils.types import Sample

__all__ = ["VAL_BASE", "generate_rollout", "is_val"]

logger = logging.getLogger(__name__)

VAL_BASE = 1_000_000


def is_val(rollout_id: int) -> bool:
    return rollout_id >= VAL_BASE


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


@cache
def _val_ids(directory: str) -> list[int]:
    return sorted(_index(directory))


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    assert not evaluation, "SFT samples have no evaluation rollout"
    if is_val(rollout_id):
        directory = os.environ["MILES_SFT_VAL_DIR"]
        ids = _val_ids(directory)
        groups = [
            [fill_sample(Sample(group_index=i, index=i, prompt=str(sid)), directory)] for i, sid in enumerate(ids)
        ]
        # Miles trims a rollout to a multiple of --global-batch-size; pad with untrained copies so none is cut
        for i in range(len(ids), -(-len(ids) // args.global_batch_size) * args.global_batch_size):
            pad = fill_sample(Sample(group_index=i, index=i, prompt=str(ids[i % len(ids)])), directory)
            pad.loss_mask = [0] * len(pad.loss_mask)
            groups.append([pad])
        return groups
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
