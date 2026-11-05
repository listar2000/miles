"""allenai/tmax-sft (OpenAI messages, one ``bash`` tool) to SFT records, with whole tasks held out.

Records keep each row's messages. Replies are repaired where the fix is unambiguous; the rest stay in the context
untrained (``steps[].train = false``):

- no ``bash`` call (the source harness answered with a "Format error" turn), a call to another tool, or chat control
  tokens: untrained;
- a ``</think>`` left in ``content`` after one THOUGHT: removed; after several: untrained;
- ``</parameter></function></tool_call>`` closing the reasoning of a parsed call: removed; other tool markup: untrained.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from miles_ext.sft.connectors.base import Dropped, Item, write_records

_SCAFFOLD_TAIL = re.compile(r"(\s*</(parameter|function|tool_call)>)+\s*$")
_CONTROL = re.compile(r"<\|(im_start|im_end|endoftext)\|>")
DATASET = "allenai/tmax-sft"
REVISION = "9d6ab7471ffa2884b2ef0cd2b7e5e22a027ff1b4"
DATA_FILE = "skill_tax_20260505_2.2k_combined_balanced_thinking_only_success/train-00000-of-00001.parquet"
HELD_OUT_TASKS = 40


def _repair_reply(message: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """The reply with its text repaired, and why it stays untrained (None: trained)."""
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    calls = message.get("tool_calls") or []
    if not calls:
        return message, "no_tool_call"
    if any((c.get("function") or {}).get("name") != "bash" for c in calls):
        return message, "not_bash"
    if _CONTROL.search(content) or _CONTROL.search(reasoning):
        return message, "control_token"
    if "</think>" in content or "<think>" in content:
        if content.count("</think>") != 1 or "<think>" in content or content.count("THOUGHT:") > 1:
            return message, "ambiguous_think_in_content"
        before, _, after = content.partition("</think>")
        if after.strip():
            return message, "ambiguous_think_in_content"
        content = before.rstrip() + "\n\n"
    if "<think>" in reasoning or "</think>" in reasoning:
        return message, "think_in_reasoning"
    if _SCAFFOLD_TAIL.search(reasoning):
        reasoning = _SCAFFOLD_TAIL.sub("", reasoning).rstrip() + "\n"
        reasoning = "" if not reasoning.strip() else reasoning
    if re.search(r"</?(tool_call|function=|parameter=)", reasoning + content):
        return message, "tool_markup_left"
    return {**message, "content": content, "reasoning_content": reasoning}, None


def _message(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a source message to record format: tool results get their one ``tool_call_id``, tool arguments become JSON strings."""
    role = raw["role"]
    out: dict[str, Any] = {"role": role, "content": raw.get("content") or ""}
    if role == "assistant":
        out["reasoning_content"] = raw.get("reasoning_content") or ""
        calls = raw.get("tool_calls") or []
        if calls:
            out["tool_calls"] = [
                {
                    "id": c.get("id"),
                    "type": "function",
                    "function": {
                        "name": c["function"]["name"],
                        "arguments": _arguments(c["function"].get("arguments")),
                    },
                }
                for c in calls
            ]
    elif role == "tool":
        ids = raw.get("tool_call_ids") or ([raw["tool_call_id"]] if raw.get("tool_call_id") else [])
        if ids:
            out["tool_call_id"] = ids[0]
    return out


def _arguments(arguments: Any) -> str:
    """Tool-call arguments as a JSON string: tmax-sft stores objects, a Vanillux2Agent trajectory stores strings."""
    return arguments if isinstance(arguments, str) else json.dumps(arguments or {}, ensure_ascii=False)


def convert_row(row: dict[str, Any], counts: dict[str, int] | None = None) -> Item:
    meta = row.get("metadata") or {}
    source = str(meta.get("trial_name") or meta.get("run_id") or "?")
    raw_messages = list(row.get("messages") or [])
    if len(raw_messages) < 3 or raw_messages[0]["role"] != "system" or raw_messages[1]["role"] != "user":
        return Dropped(source, "unexpected_shape")
    messages: list[dict[str, Any]] = []
    steps = []
    for index, raw in enumerate(raw_messages):
        message = _message(raw)
        if message["role"] == "assistant":
            message, untrained = _repair_reply(message)
            steps.append({"context": list(range(index)), "completion": index, "train": untrained is None})
            if counts is not None:
                counts[untrained or "trained"] = counts.get(untrained or "trained", 0) + 1
        messages.append(message)
    if not any(s["train"] for s in steps):
        return Dropped(source, "no_trained_reply")
    return {
        "messages": messages,
        "tools": json.loads(row["tools"]),
        "steps": steps,
        "metadata": {
            "trajectory": source,
            "task": meta.get("task"),
            "source_model": meta.get("source_model"),
            "num_turns": meta.get("num_turns"),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path, help="directory for success.jsonl and success.val.jsonl")
    args = parser.parse_args(argv)

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    parquet = hf_hub_download(DATASET, DATA_FILE, repo_type="dataset", revision=REVISION)
    rows = pq.read_table(parquet).to_pylist()
    tasks = sorted({(r.get("metadata") or {}).get("task") for r in rows} - {None})
    held_out = set(random.Random(0).sample(tasks, HELD_OUT_TASKS))
    counts: dict[str, int] = {}
    train: list[Item] = []
    val: list[Item] = []
    for row in rows:
        (val if (row.get("metadata") or {}).get("task") in held_out else train).append(convert_row(row, counts))
    args.output.mkdir(parents=True, exist_ok=True)
    report = write_records(train, args.output / "success.jsonl")
    val_report = write_records(val, args.output / "success.val.jsonl")
    print(
        json.dumps(
            {"train": report.records, "val": val_report.records, "dropped": dict(report.dropped), "replies": counts}
        )
    )


if __name__ == "__main__":
    main()
