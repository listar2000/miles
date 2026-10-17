"""ContextExitAgent ends a full-window run "ContextExhausted" instead of nagging it, over a stand-in mini-swe-agent."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "harbor" / "context_exit.py"


class InterruptAgentFlow(Exception):
    def __init__(self, *messages):
        self.messages = messages
        super().__init__()


class LimitsExceeded(InterruptAgentFlow):
    pass


class FormatError(InterruptAgentFlow):
    pass


class InteractiveAgent:  # mini-swe-agent 2.4.6: query raises FormatError before appending the reply
    def __init__(self, model, messages):
        self.model, self.messages = model, messages

    def query(self):
        message = self.model.query(self.messages)
        self.messages.append(message)
        return message


@pytest.fixture
def ce(monkeypatch):
    stubs = {
        "minisweagent": types.SimpleNamespace(),
        "minisweagent.agents": types.SimpleNamespace(),
        "minisweagent.agents.interactive": types.SimpleNamespace(InteractiveAgent=InteractiveAgent),
        "minisweagent.exceptions": types.SimpleNamespace(FormatError=FormatError, LimitsExceeded=LimitsExceeded),
    }
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)
    spec = importlib.util.spec_from_file_location("miles_mswea_context_under_test", _SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv(module.ENV_TOKENS, "1000")
    monkeypatch.setenv(module.ENV_MIN_REPLY, "100")
    return module


class _Usage:  # litellm's in-memory response objects use attribute access
    def __init__(self, prompt, completion):
        self.prompt_tokens, self.completion_tokens = prompt, completion


def _turn(prompt, completion):
    return {"role": "assistant", "content": "x", "extra": {"response": {"usage": _Usage(prompt, completion)}}}


def _nag(prompt, completion, content="reached the output token limit"):
    usage = {"prompt_tokens": prompt, "completion_tokens": completion}
    return {"role": "user", "content": content, "extra": {"response": {"usage": usage}}}


class _Model:
    def __init__(self, result):
        self.result, self.calls = result, 0

    def query(self, messages):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_room_left_queries_normally(ce):
    model = _Model(_turn(600, 50))
    ce.ContextExitAgent(model, [_turn(400, 100), {"role": "tool", "content": "y" * 300}]).query()
    assert model.calls == 1


def test_full_window_exits_before_the_call(ce):
    model = _Model(_turn(0, 0))
    agent = ce.ContextExitAgent(model, [_turn(600, 100), {"role": "tool", "content": "y" * 900}])
    with pytest.raises(LimitsExceeded) as e:
        agent.query()
    assert e.value.messages[0]["extra"]["exit_status"] == ce.EXIT_STATUS
    assert model.calls == 0


def test_cut_off_turn_exits_instead_of_nagging(ce):
    agent = ce.ContextExitAgent(_Model(FormatError(_nag(800, 190))), [_turn(10, 10)])
    with pytest.raises(LimitsExceeded):
        agent.query()
    assert len(agent.messages) == 1


def test_call_after_a_nag_at_the_wall_exits(ce):
    model = _Model(_turn(0, 0))
    agent = ce.ContextExitAgent(model, [_turn(400, 100), {"role": "tool", "content": "y"}, _nag(950, 0)])
    with pytest.raises(LimitsExceeded):
        agent.query()
    assert model.calls == 0


def test_discarded_reply_of_a_nag_does_not_count(ce):
    model = _Model(_turn(0, 0))
    ce.ContextExitAgent(model, [_turn(300, 50), {"role": "tool", "content": "y"}, _nag(400, 600)]).query()
    assert model.calls == 1


def test_ordinary_format_error_still_nags(ce):
    agent = ce.ContextExitAgent(_Model(FormatError(_nag(100, 20, "no tool call"))), [_turn(10, 10)])
    with pytest.raises(FormatError):
        agent.query()


def test_unset_limit_does_nothing(ce, monkeypatch):
    monkeypatch.delenv(ce.ENV_TOKENS)
    model = _Model(_turn(0, 0))
    ce.ContextExitAgent(model, [_turn(10**6, 10**6)]).query()
    assert model.calls == 1
