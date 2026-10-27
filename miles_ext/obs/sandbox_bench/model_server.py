"""A scripted OpenAI-compatible model: each chat completion of a session returns that session's next command.

It stands where Miles' session server stands (``http://<host>:<port>/sessions/<id>``), behind the real session proxy
and tunnel, so the in-sandbox mini-swe-agent cannot tell it from training except by speed. Every reply is one ``bash``
tool call; after the script, the reply is mini-swe-agent's submit command.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field

from aiohttp import web


SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"  # mini-swe-agent's finish command (its mini.yaml)


@dataclass
class Script:
    commands: list[str]
    think_s: list[float] | None = None  # seconds to wait before each reply; None: answer at once
    calls: int = field(default=0, init=False)

    def next(self) -> tuple[str, float]:
        i = self.calls
        self.calls += 1
        if i >= len(self.commands):
            return SUBMIT, 0.0
        return self.commands[i], (self.think_s[i] if self.think_s and i < len(self.think_s) else 0.0)


def completion(command: str, step: int) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": f"step {step}",
                    "tool_calls": [
                        {
                            "id": f"call_{step}",
                            "type": "function",
                            "function": {"name": "bash", "arguments": json.dumps({"command": command})},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def make_app(scripts: dict[str, Script], log: list[dict] | None = None) -> web.Application:
    """``scripts``: session id -> its script; the caller adds sessions before their trials start.
    ``log``: a list each call's server time is appended to."""

    async def chat(request: web.Request) -> web.Response:
        t0, mono = time.time(), time.monotonic()
        key = request.match_info["session"]
        script = scripts.get(key)
        if script is None:
            raise web.HTTPNotFound(text=f"no script for session {key!r}")
        await request.read()
        command, think = script.next()
        if think > 0:
            await asyncio.sleep(think)
        response = web.json_response(completion(command, script.calls))
        if log is not None:  # the caller writes it, off the event loop
            log.append({"t0": t0, "dt": time.monotonic() - mono, "session": key, "step": script.calls})
        return response

    async def health(request):
        return web.Response(text="ok")

    app = web.Application(client_max_size=256 * 1024**2)
    app.router.add_post("/sessions/{session}/v1/chat/completions", chat)
    app.router.add_get("/healthz", health)
    return app
