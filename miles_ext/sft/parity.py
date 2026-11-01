"""Parity: does a rendered SFT sample match what the session server recorded in a real rollout?

``compare`` takes a served sample (all its tokens and the loss mask over its response,
as a session-server ``Sample`` stores them) and the render of the same record, and
reports whether tokens and masks are equal and where they first differ. Pairing a
rollout's samples with records depends on the harness and belongs to the adapter that
builds the records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from miles_ext.sft.render import RenderedSample

WINDOW = 20  # tokens decoded on each side of the first difference


@dataclass(frozen=True)
class ParityReport:
    tokens_equal: bool
    mask_equal: bool
    rendered_length: int
    served_length: int
    first_token_diff: int | None
    mask_diff_count: int
    rendered_window: str = ""
    served_window: str = ""


def full_loss_mask(tokens: list[int], response_length: int, loss_mask: list[int]) -> list[int]:
    """A served sample's response mask, extended with zeros over its prompt."""
    return [0] * (len(tokens) - response_length) + list(loss_mask)


def compare(
    rendered: RenderedSample,
    *,
    served_tokens: list[int],
    served_loss_mask: list[int],
    tokenizer: Any = None,
) -> ParityReport:
    """Compare a render with a served sample; *served_loss_mask* covers all served tokens."""
    first = next(
        (i for i, (a, b) in enumerate(zip(rendered.token_ids, served_tokens, strict=False)) if a != b),
        None if len(rendered.token_ids) == len(served_tokens) else min(len(rendered.token_ids), len(served_tokens)),
    )
    windows = ("", "")
    if first is not None and tokenizer is not None:
        lo, hi = max(first - WINDOW, 0), first + WINDOW
        windows = (tokenizer.decode(rendered.token_ids[lo:hi]), tokenizer.decode(served_tokens[lo:hi]))
    mask_diff = sum(a != b for a, b in zip(rendered.loss_mask, served_loss_mask, strict=False))
    mask_diff += abs(len(rendered.loss_mask) - len(served_loss_mask))
    return ParityReport(
        tokens_equal=first is None,
        mask_equal=mask_diff == 0,
        rendered_length=len(rendered.token_ids),
        served_length=len(served_tokens),
        first_token_diff=first,
        mask_diff_count=mask_diff,
        rendered_window=windows[0],
        served_window=windows[1],
    )
