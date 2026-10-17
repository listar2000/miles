"""The context-exit patch ends a full-window run "ContextExhausted" instead of nagging it three times:

python -m pytest miles_ext/tests/test_context_exit.py

The patch is applied to a stand-in minisweagent package whose ``query`` is mini-swe-agent 2.4.6's, after
Harbor's step-timing patch (the order the sandbox install runs them in).
"""

import importlib
import subprocess
import sys
import textwrap

import pytest

from miles_ext.harbor import context_exit

_EXCEPTIONS = '''
class InterruptAgentFlow(Exception):
    def __init__(self, *messages):
        self.messages = messages
        super().__init__()


class LimitsExceeded(InterruptAgentFlow):
    pass


class FormatError(InterruptAgentFlow):
    pass
'''

# mini-swe-agent 2.4.6 DefaultAgent.query, with Harbor's step-timing patch applied
_DEFAULT = '''
import time

from minisweagent.exceptions import FormatError, InterruptAgentFlow, LimitsExceeded


class DefaultAgent:
    def __init__(self, model, messages):
        self.model, self.messages, self.n_calls = model, messages, 0

    def query(self):
        self.n_calls += 1
        _harbor_t0 = time.time()
        message = self.model.query(self.messages)
        message.setdefault("extra", {})["harbor_llm_wait_msec"] = (time.time() - _harbor_t0) * 1000
        self.messages.append(message)
        return message
'''


class _Usage:  # litellm's in-memory response objects use attribute access
    def __init__(self, prompt, completion):
        self.prompt_tokens, self.completion_tokens = prompt, completion


def _turn(prompt, completion, content="x"):
    return {"role": "assistant", "content": content, "extra": {"response": {"usage": _Usage(prompt, completion)}}}


@pytest.fixture
def agent_module(tmp_path, monkeypatch):
    pkg = tmp_path / "minisweagent"
    (pkg / "agents").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "agents" / "__init__.py").write_text("")
    (pkg / "exceptions.py").write_text(textwrap.dedent(_EXCEPTIONS))
    (pkg / "agents" / "default.py").write_text(textwrap.dedent(_DEFAULT))
    env = {"PYTHONPATH": str(tmp_path)}
    for _ in range(2):  # idempotent
        out = subprocess.run([sys.executable, "-"], input=context_exit.PATCH, text=True, env=env,
                             capture_output=True, check=True).stdout
    assert "already applied" in out
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [m for m in sys.modules if m.startswith("minisweagent")]:
        del sys.modules[name]
    monkeypatch.setenv(context_exit.ENV_TOKENS, "1000")
    monkeypatch.setenv(context_exit.ENV_MIN_REPLY, "100")
    yield importlib.import_module("minisweagent.agents.default"), importlib.import_module("minisweagent.exceptions")
    for name in [m for m in sys.modules if m.startswith("minisweagent")]:
        del sys.modules[name]


class _Model:
    def __init__(self, result):
        self.result, self.calls = result, 0

    def query(self, messages):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_room_left_queries_normally(agent_module):
    default, _ = agent_module
    model = _Model(_turn(600, 50))
    agent = default.DefaultAgent(model, [{"role": "user", "content": "task"}, _turn(400, 100), {"role": "tool", "content": "y" * 300}])
    assert agent.query()["extra"]["harbor_llm_wait_msec"] >= 0
    assert model.calls == 1


def test_full_window_exits_before_the_call(agent_module):
    default, exc = agent_module
    model = _Model(_turn(0, 0))
    # 700 + 900 chars / 2 + 100 reserve > 1000
    agent = default.DefaultAgent(model, [_turn(600, 100), {"role": "tool", "content": "y" * 900}])
    with pytest.raises(exc.LimitsExceeded) as e:
        agent.query()
    assert e.value.messages[0]["extra"]["exit_status"] == context_exit.EXIT_STATUS
    assert model.calls == 0 and agent.n_calls == 0


def test_cut_off_turn_exits_instead_of_nagging(agent_module):
    default, exc = agent_module
    nag = {"role": "user", "content": "reached the output token limit", "extra": {"response": {"usage": {"prompt_tokens": 800, "completion_tokens": 190}}}}
    agent = default.DefaultAgent(_Model(exc.FormatError(nag)), [_turn(10, 10)])
    with pytest.raises(exc.LimitsExceeded) as e:
        agent.query()
    assert e.value.messages[0]["content"] == context_exit.EXIT_STATUS
    assert len(agent.messages) == 1  # the cut-off turn is not appended


def test_ordinary_format_error_still_nags(agent_module):
    default, exc = agent_module
    nag = {"role": "user", "content": "no tool call", "extra": {"response": {"usage": {"prompt_tokens": 100, "completion_tokens": 20}}}}
    agent = default.DefaultAgent(_Model(exc.FormatError(nag)), [_turn(10, 10)])
    with pytest.raises(exc.FormatError):
        agent.query()


def test_unset_limit_does_nothing(agent_module, monkeypatch):
    default, _ = agent_module
    monkeypatch.delenv(context_exit.ENV_TOKENS)
    model = _Model(_turn(0, 0))
    default.DefaultAgent(model, [_turn(10**6, 10**6)]).query()
    assert model.calls == 1
