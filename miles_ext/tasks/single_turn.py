"""Request one session-server completion and grade it with the prompt row's verifier."""

import functools
import importlib
import logging

import httpx

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_S = 3600


@functools.cache
def _verifier(task: str):
    """metadata.task is the dotted path of the row's grade(reply, label) -> (reward 0/1, kind)."""
    module, _, name = task.rpartition(".")
    verifier = getattr(importlib.import_module(module), name)
    if not callable(verifier):
        raise TypeError(f"{task} is not callable")
    return verifier


def check_row(row: dict) -> None:
    """A prompt row this agent can grade: a known verifier and a label."""
    metadata = row.get("metadata") or {}
    task = metadata.get("task")
    try:
        _verifier(task)
    except (AttributeError, ImportError, TypeError, ValueError) as e:
        raise ValueError(f"metadata.task must be the dotted path of a verifier, got {task!r}: {e}") from e
    if not isinstance(metadata.get("label"), str):
        raise ValueError("metadata.label (the expected answer) is required")


def _messages(prompt) -> list[dict]:
    return prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]


async def run(base_url, prompt, request_kwargs, metadata, **kwargs):
    check_row({"metadata": metadata})
    payload = {**request_kwargs, "messages": _messages(prompt), "stream": False}
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
        resp = await client.post(f"{base_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
    choice = resp.json()["choices"][0]
    reward, kind = _verifier(metadata["task"])(choice["message"].get("content") or "", metadata["label"])
    return {"reward": float(reward), "grade": kind, "finish_reason": choice.get("finish_reason")}
