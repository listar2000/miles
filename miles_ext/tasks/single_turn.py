"""Request one session-server completion and grade it with the prompt row's verifier."""

import logging

import httpx

from miles_ext.tasks import stackvm

logger = logging.getLogger(__name__)

# metadata.task -> grade(reply, label) -> (reward 0/1, kind)
VERIFIERS = {stackvm.TASK: stackvm.grade}
REQUEST_TIMEOUT_S = 3600


def check_row(row: dict) -> None:
    """A prompt row this agent can grade: a known verifier and a label."""
    metadata = row.get("metadata") or {}
    if metadata.get("task") not in VERIFIERS:
        raise ValueError(f"metadata.task must be one of {sorted(VERIFIERS)}, got {metadata.get('task')!r}")
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
    reward, kind = VERIFIERS[metadata["task"]](choice["message"].get("content") or "", metadata["label"])
    return {"reward": float(reward), "grade": kind, "finish_reason": choice.get("finish_reason")}
