"""Render SFT records offline (CPU) into token shards that ``miles_ext.sft.rollout`` serves.

    python -m miles_ext.sft.prepare --input data.jsonl --output out/ \\
        --hf-checkpoint /path/to/model --tito-model qwennext \\
        [--thinking-retention all|template] [--postprocess miles_ext.sft.postprocess.split_inline_think ...] \\
        [--max-tokens 65536]

Output directory:

- ``prompts.jsonl``: one ``{"text": "<sample id>"}`` row per sample, the file for ``--prompt-data``.
- ``samples.jsonl``: per sample, its record line, shard, offset, length and trained tokens.
- ``shards/NNNNN.tokens.npy`` (int32) and ``shards/NNNNN.mask.npy`` (uint8): samples back to back.
- ``manifest.json``: the settings, counts, and the most common reasons records were skipped.

A record that fails validation or rendering is skipped and counted; so is a sample longer
than ``--max-tokens`` (it is dropped, not cut) or one with no trained token.
"""

from __future__ import annotations

import argparse
import collections
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from miles_ext.sft.postprocess import Processor, apply_processors, load_processors
from miles_ext.sft.record import parse_record
from miles_ext.sft.render import THINKING_RETENTION, RenderedSample, SftRenderer, load_renderer

SHARD_TOKENS = 1 << 27  # about 512 MB of int32 tokens per shard
TOP_ERRORS = 10


@dataclass
class _Counts:
    records: int = 0
    records_skipped: int = 0
    records_split: int = 0
    samples: int = 0
    samples_too_long: int = 0
    samples_untrained: int = 0
    tokens: int = 0
    trained_tokens: int = 0
    errors: collections.Counter = field(default_factory=collections.Counter)


class _ShardWriter:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shard = 0
        self.tokens: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.offset = 0

    def add(self, sample: RenderedSample) -> tuple[int, int]:
        if self.offset and self.offset + len(sample.token_ids) > SHARD_TOKENS:
            self.flush()
        where = (self.shard, self.offset)
        self.tokens.append(np.asarray(sample.token_ids, dtype=np.int32))
        self.masks.append(np.asarray(sample.loss_mask, dtype=np.uint8))
        self.offset += len(sample.token_ids)
        return where

    def flush(self) -> None:
        if not self.tokens:
            return
        np.save(self.directory / f"{self.shard:05d}.tokens.npy", np.concatenate(self.tokens))
        np.save(self.directory / f"{self.shard:05d}.mask.npy", np.concatenate(self.masks))
        self.shard, self.offset, self.tokens, self.masks = self.shard + 1, 0, [], []


def _read_records(paths: list[str]) -> Iterator[tuple[str, dict]]:
    for path in paths:
        with open(path) as f:
            for line_number, line in enumerate(f, start=1):
                if line.strip():
                    yield f"{path}:{line_number}", json.loads(line)


def prepare(
    *,
    inputs: list[str],
    output: Path,
    renderer: SftRenderer,
    processors: list[Processor],
    max_tokens: int | None,
) -> _Counts:
    """Render every record of *inputs* into *output*; return the counts written to the manifest."""
    counts = _Counts()
    writer = _ShardWriter(output / "shards")
    with open(output / "samples.jsonl", "w") as index, open(output / "prompts.jsonl", "w") as prompts:
        for source, raw in _read_records(inputs):
            counts.records += 1
            try:
                samples = renderer.render(parse_record(apply_processors(raw, processors)))
            except Exception as exc:  # one bad record must not stop a dataset build
                counts.records_skipped += 1
                counts.errors[f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"] += 1
                continue
            counts.records_split += len(samples) > 1
            for sample in samples:
                trained = sum(sample.loss_mask)
                if not trained:
                    counts.samples_untrained += 1
                    continue
                if max_tokens is not None and len(sample.token_ids) > max_tokens:
                    counts.samples_too_long += 1
                    continue
                shard, offset = writer.add(sample)
                row = dict(
                    id=counts.samples,
                    source=source,
                    shard=shard,
                    offset=offset,
                    length=len(sample.token_ids),
                    trained_tokens=trained,
                    trained_steps=list(sample.trained_steps),
                )
                index.write(json.dumps(row) + "\n")
                prompts.write(json.dumps({"text": str(counts.samples)}) + "\n")
                counts.samples += 1
                counts.tokens += len(sample.token_ids)
                counts.trained_tokens += trained
    writer.flush()
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", action="append", required=True, help="a JSONL file of SFT records (repeatable)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--tito-model", required=True, help="the --tito-model the model is served with")
    parser.add_argument("--thinking-retention", default="all", choices=THINKING_RETENTION)
    parser.add_argument("--postprocess", action="append", default=[], help="a processor's dotted path (repeatable)")
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    renderer = load_renderer(args.hf_checkpoint, args.tito_model, thinking_retention=args.thinking_retention)
    counts = prepare(
        inputs=args.input,
        output=args.output,
        renderer=renderer,
        processors=load_processors(args.postprocess),
        max_tokens=args.max_tokens,
    )
    manifest = dict(
        inputs=args.input,
        hf_checkpoint=args.hf_checkpoint,
        tito_model=args.tito_model,
        thinking_retention=args.thinking_retention,
        postprocess=args.postprocess,
        max_tokens=args.max_tokens,
        **{k: v for k, v in vars(counts).items() if k != "errors"},
        top_errors=counts.errors.most_common(TOP_ERRORS),
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "top_errors"}, indent=1))


if __name__ == "__main__":
    main()
