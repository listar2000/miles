"""Record post-processors, run in order before validation; each returns a new dict and leaves its input unchanged."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from miles.utils.function_registry import load_function

Processor = Callable[[dict[str, Any]], dict[str, Any]]

_INLINE_THINK = re.compile(r"^\s*(?:<think>)?(.*?)</think>\s*(.*)$", re.DOTALL)


def load_processors(paths: Sequence[str]) -> list[Processor]:
    return [load_function(path, sync_required=True) for path in paths]


def apply_processors(record: dict[str, Any], processors: Sequence[Processor]) -> dict[str, Any]:
    for processor in processors:
        record = processor(record)
    return record


def _map_messages(record: dict[str, Any], edit: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    out = copy.deepcopy(record)
    out["messages"] = [edit(message) for message in out.get("messages") or []]
    return out


def normalize_openai_messages(record: dict[str, Any]) -> dict[str, Any]:
    """Common OpenAI-export variations to the record format: ``content: null`` becomes ``""``,
    ``tool_calls`` given as a JSON string is parsed, tool-call arguments given as objects become
    JSON strings, and an empty ``tool_calls`` list is dropped."""

    def edit(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("content") is None:
            message["content"] = ""
        calls = message.get("tool_calls")
        if isinstance(calls, str):
            calls = json.loads(calls) if calls.strip() else None
        if not calls:
            message.pop("tool_calls", None)
            return message
        for call in calls:
            function = call.get("function") or {}
            if not isinstance(function.get("arguments"), str):
                function["arguments"] = json.dumps(function.get("arguments") or {}, ensure_ascii=False)
        message["tool_calls"] = calls
        return message

    return _map_messages(record, edit)


def split_inline_think(record: dict[str, Any]) -> dict[str, Any]:
    """Move reasoning written inline as ``<think>...</think>`` in assistant content into
    ``reasoning_content``. Messages that already have ``reasoning_content`` are left alone."""

    def edit(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("role") != "assistant" or message.get("reasoning_content") is not None:
            return message
        match = _INLINE_THINK.match(message.get("content") or "")
        if match is not None:
            message["reasoning_content"] = match.group(1).strip("\n")
            message["content"] = match.group(2)
        return message

    return _map_messages(record, edit)


def strip_assistant_trailing_whitespace(record: dict[str, Any]) -> dict[str, Any]:
    """Trailing whitespace off assistant content and reasoning. Raw ``completion_token_ids``
    are dropped from an edited message, since they no longer match its text."""

    def edit(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("role") != "assistant":
            return message
        before = (message.get("content"), message.get("reasoning_content"))
        for key in ("content", "reasoning_content"):
            if isinstance(message.get(key), str):
                message[key] = message[key].rstrip()
        if (message.get("content"), message.get("reasoning_content")) != before:
            message.pop("completion_token_ids", None)
        return message

    return _map_messages(record, edit)
