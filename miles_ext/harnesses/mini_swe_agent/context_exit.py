"""mini-swe-agent agent class that ends a run "ContextExhausted" when the context window is full.

Self-contained (imports only mini-swe-agent): ``SandboxMiniSweAgent.install`` copies it into the sandbox as
``AGENT_MODULE`` (fork/patches/ext-001-context-exit.md).
"""

import os

from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.exceptions import FormatError, LimitsExceeded

# Harbor redacts agent env values whose key contains KEY, TOKEN, AUTH, ... in trial files: keep these names clear of them
ENV_LIMIT = "MSWEA_CONTEXT_LIMIT"
ENV_MIN_REPLY = "MSWEA_CONTEXT_MIN_REPLY"
EXIT_STATUS = "ContextExhausted"
AGENT_MODULE = "miles_mswea_context"
AGENT_CLASS = f"{AGENT_MODULE}.ContextExitAgent"

_DEFAULT_MIN_REPLY = 4096
# a reply that ran into the window ends within this many tokens of it (sizing: patch page)
_WALL_MARGIN = 64


def _get(obj, key):
    # in memory a response is litellm's ModelResponse (attribute access); in a saved trajectory, a dict
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _usage(message: dict) -> tuple[int, int]:
    usage = _get(_get(message.get("extra") or {}, "response") or {}, "usage") or {}
    return _get(usage, "prompt_tokens") or 0, _get(usage, "completion_tokens") or 0


def _chars(messages) -> int:
    return sum(len(str(m.get("content") or "")) for m in messages)


def window_full(messages: list[dict], limit: int, min_reply: int) -> bool:
    """No room left for a useful reply: tokens in context at the last call + the messages since + the reserve."""
    for i in range(len(messages) - 1, -1, -1):
        prompt, completion = _usage(messages[i])
        if not (prompt or completion):
            continue
        if messages[i].get("role") == "assistant":
            in_context = prompt + completion
        else:
            # a format-error nag carries its reply's usage, but the reply was discarded; the nag text replaced it
            in_context = prompt + _chars(messages[i : i + 1]) // 2
        return in_context + _chars(messages[i + 1 :]) // 2 + min_reply > limit
    return False


def reply_hit_window(messages, limit: int) -> bool:
    return any(sum(_usage(m)) >= limit - _WALL_MARGIN for m in messages)


def _env_int(name: str) -> int:
    try:
        return int(os.environ.get(name) or 0)
    except ValueError:
        return 0


def _exit() -> LimitsExceeded:
    return LimitsExceeded(
        {"role": "exit", "content": EXIT_STATUS, "extra": {"exit_status": EXIT_STATUS, "submission": ""}}
    )


class ContextExitAgent(InteractiveAgent):
    def query(self) -> dict:
        limit = _env_int(ENV_LIMIT)
        if limit and window_full(self.messages, limit, _env_int(ENV_MIN_REPLY) or _DEFAULT_MIN_REPLY):
            raise _exit()
        try:
            return super().query()
        except FormatError as e:
            if limit and reply_hit_window(e.messages, limit):
                raise _exit() from e
            raise
