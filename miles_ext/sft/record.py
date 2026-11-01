"""Validate SFT records: OpenAI messages used as a store, and the context of each model call."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

ROLES = frozenset({"system", "user", "assistant", "tool"})
RECORD_KEYS = frozenset({"messages", "tools", "chat_template_kwargs", "steps", "metadata"})
STEP_KEYS = frozenset({"context", "completion", "train"})


class RecordError(ValueError):
    """The record does not follow the SFT record format."""


@dataclass(frozen=True)
class Step:
    """One model call: the messages the harness sent (indices into the store) and the reply."""

    context: tuple[int, ...]
    completion: int
    train: bool = True


@dataclass(frozen=True)
class SftRecord:
    messages: tuple[dict[str, Any], ...]
    steps: tuple[Step, ...]
    tools: list[dict[str, Any]] | None = None
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def context_messages(self, step: Step) -> list[dict[str, Any]]:
        return [self.messages[i] for i in step.context]


def parse_record(raw: dict[str, Any]) -> SftRecord:
    """Validate *raw* and return it as an ``SftRecord``; raise ``RecordError`` otherwise."""
    if not isinstance(raw, dict):
        raise RecordError("a record is a JSON object")
    unknown = set(raw) - RECORD_KEYS
    if unknown:
        raise RecordError(f"unknown record keys {sorted(unknown)}")
    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RecordError("messages must be a non-empty list")
    for index, message in enumerate(messages):
        _check_message(index, message)
    raw_steps = raw.get("steps")
    steps = _default_steps(messages) if raw_steps is None else _parse_steps(raw_steps, messages)
    tools = raw.get("tools") or None
    if tools is not None and not (isinstance(tools, list) and all(isinstance(t, dict) for t in tools)):
        raise RecordError("tools must be a list of OpenAI tool objects")
    kwargs = raw.get("chat_template_kwargs") or {}
    metadata = raw.get("metadata") or {}
    if not isinstance(kwargs, dict) or not isinstance(metadata, dict):
        raise RecordError("chat_template_kwargs and metadata must be objects")
    return SftRecord(
        messages=tuple(messages),
        steps=steps,
        tools=tools,
        chat_template_kwargs=dict(kwargs),
        metadata=dict(metadata),
    )


def _check_message(index: int, message: Any) -> None:
    where = f"messages[{index}]"
    if not isinstance(message, dict):
        raise RecordError(f"{where} is not an object")
    role = message.get("role")
    if role not in ROLES:
        raise RecordError(f"{where}: role {role!r} is not one of {sorted(ROLES)}")
    if not isinstance(message.get("content"), str):
        raise RecordError(f"{where}: content must be a string")
    if role != "assistant":
        return
    reasoning = message.get("reasoning_content")
    if reasoning is not None and not isinstance(reasoning, str):
        raise RecordError(f"{where}: reasoning_content must be a string")
    for call in message.get("tool_calls") or []:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise RecordError(f"{where}: a tool call needs function.name")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise RecordError(f"{where}: tool-call arguments must be a JSON string")
        try:
            json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise RecordError(f"{where}: tool-call arguments are not JSON ({exc})") from exc
    ids = message.get("completion_token_ids")
    if ids is not None and not (isinstance(ids, list) and ids and all(type(i) is int for i in ids)):
        raise RecordError(f"{where}: completion_token_ids must be a non-empty list of ints")


def _default_steps(messages: list[dict[str, Any]]) -> tuple[Step, ...]:
    """An append-only harness: every assistant message is a trained step after everything before it."""
    steps = tuple(
        Step(context=tuple(range(i)), completion=i)
        for i, message in enumerate(messages)
        if message["role"] == "assistant" and i > 0
    )
    if not steps:
        raise RecordError("no assistant message after a prompt")
    return steps


def _parse_steps(raw_steps: Any, messages: list[dict[str, Any]]) -> tuple[Step, ...]:
    if not isinstance(raw_steps, list) or not raw_steps:
        raise RecordError("steps must be a non-empty list")
    steps = []
    for k, raw in enumerate(raw_steps):
        where = f"steps[{k}]"
        if not isinstance(raw, dict) or set(raw) - STEP_KEYS:
            raise RecordError(f"{where}: a step has only {sorted(STEP_KEYS)}")
        context, completion, train = raw.get("context"), raw.get("completion"), raw.get("train", True)
        if not isinstance(context, list) or not context or not all(type(i) is int for i in context):
            raise RecordError(f"{where}: context must be a non-empty list of message indices")
        if type(completion) is not int or not isinstance(train, bool):
            raise RecordError(f"{where}: completion is a message index and train a boolean")
        for i in [*context, completion]:
            if not 0 <= i < len(messages):
                raise RecordError(f"{where}: index {i} is outside messages")
        if messages[completion]["role"] != "assistant":
            raise RecordError(f"{where}: completion {completion} is not an assistant message")
        if completion in context:
            raise RecordError(f"{where}: the completion is also in its context")
        steps.append(Step(context=tuple(context), completion=completion, train=train))
    return tuple(steps)
