"""Record processors for this experiment's ``data.postprocess``."""

from __future__ import annotations

from typing import Any


def untrain_no_reasoning(record: dict[str, Any]) -> dict[str, Any]:
    """Keep replies without reasoning in the context but out of the loss (decision B2)."""
    messages = record["messages"]
    steps = record.get("steps") or [
        {"context": list(range(i)), "completion": i, "train": True}
        for i, m in enumerate(messages)
        if m["role"] == "assistant" and i > 0
    ]
    for step in steps:
        if not (messages[step["completion"]].get("reasoning_content") or "").strip():
            step["train"] = False
    return {**record, "steps": steps}
