"""End a mini-swe-agent run cleanly at the context window ("ContextExhausted").

Without this, a trajectory that fills the window keeps going: the engine truncates the next turn's
``max_new_tokens`` to what is left, the turn ends ``finish_reason=length`` with no tool call, mini-swe-agent
answers with a format-error nag, the next turn has even less room (often 0 tokens), and after three such
turns the run exits ``RepeatedFormatError``. In dsv41-tb4-overfit16-4 that was 55% of trajectories.

``PATCH`` is a script run with the sandbox's mini-swe-agent interpreter at install time (like Harbor's own
output-cap and step-timing patches). It edits ``minisweagent/agents/default.py`` so that ``query``:

- before calling the model, exits ``ContextExhausted`` when the last call's ``prompt + completion`` tokens
  plus an estimate of the observations since (chars / 2) leave less than ``MSWEA_CONTEXT_MIN_REPLY`` tokens
  (default 4096) of ``MSWEA_CONTEXT_TOKENS``;
- when the model call fails to parse and that response filled the window (``prompt + completion`` within 64
  tokens of ``MSWEA_CONTEXT_TOKENS``; in dsv41-tb4-overfit16-4 such turns ended 2-6 tokens short), exits
  ``ContextExhausted`` instead of nagging, without appending the cut-off turn. A reply cut by the per-turn
  ``max_tokens`` far from the window still gets the nag.

The exit is an ordinary ``LimitsExceeded`` flow, so Harbor still runs the verifier and the sample is scored
normally (not aborted). With ``MSWEA_CONTEXT_TOKENS`` unset the patched code does nothing.
"""

ENV_TOKENS = "MSWEA_CONTEXT_TOKENS"
ENV_MIN_REPLY = "MSWEA_CONTEXT_MIN_REPLY"
EXIT_STATUS = "ContextExhausted"

PATCH = r'''
import pathlib

import minisweagent.agents.default as m

p = pathlib.Path(m.__file__)
s = p.read_text()
query_anchor = "        message = self.model.query(self.messages)\n"
count_anchor = "        self.n_calls += 1\n"
helpers = """

def _mswea_ctx_limit():
    import os

    try:
        return int(os.environ.get("MSWEA_CONTEXT_TOKENS") or 0)
    except ValueError:
        return 0


def _mswea_get(obj, key):
    # in memory the response is litellm's ModelResponse (attribute access); in a saved trajectory, a dict
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _mswea_usage(message):
    usage = _mswea_get(_mswea_get(message.get("extra") or {}, "response") or {}, "usage") or {}
    return (_mswea_get(usage, "prompt_tokens") or 0) + (_mswea_get(usage, "completion_tokens") or 0)


def _mswea_ctx_full(messages):
    # no room for a useful reply: last call's tokens + observations since (chars / 2) + the reply reserve.
    # The last call may be a format-error nag (role user), which carries its response's usage too.
    import os

    limit = _mswea_ctx_limit()
    if not limit:
        return False
    for i in range(len(messages) - 1, -1, -1):
        if _mswea_usage(messages[i]):
            after = sum(len(str(x.get("content") or "")) for x in messages[i + 1:])
            reserve = int(os.environ.get("MSWEA_CONTEXT_MIN_REPLY") or 4096)
            return _mswea_usage(messages[i]) + after // 2 + reserve > limit
    return False


def _mswea_filled_window(messages):
    limit = _mswea_ctx_limit()
    return bool(limit) and any(_mswea_usage(x) >= limit - 64 for x in messages)


def _mswea_ctx_exit():
    return LimitsExceeded({"role": "exit", "content": "ContextExhausted",
                           "extra": {"exit_status": "ContextExhausted", "submission": ""}})
"""
query_patch = """        try:
            message = self.model.query(self.messages)
        except FormatError as _mswea_e:
            if _mswea_filled_window(_mswea_e.messages):
                raise _mswea_ctx_exit()
            raise
"""
count_patch = """        if _mswea_ctx_full(self.messages):
            raise _mswea_ctx_exit()
        self.n_calls += 1
"""
if "_mswea_ctx_exit" in s:
    print("mini-swe-agent context-exit patch already applied")
elif query_anchor not in s or count_anchor not in s:
    print("mini-swe-agent context-exit anchors not found; NOT installed")  # fail-soft, as Harbor's own patches
else:
    s = s.replace(query_anchor, query_patch, 1).replace(count_anchor, count_patch, 1) + helpers
    p.write_text(s)
    print("mini-swe-agent context-exit patch installed")
'''
