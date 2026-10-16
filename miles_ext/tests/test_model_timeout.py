"""Model calls from both mini-swe-agent harnesses carry the trial cap as their timeout,
not the OpenAI client's 600 s default, which cut Run 3's long turns:

    python -m pytest miles_ext/tests/test_model_timeout.py
"""

from miles_ext.harbor import agent_function as af


def _no_harbor(monkeypatch):
    monkeypatch.setenv("AGENT_TRIAL_TIMEOUT", "32400")
    monkeypatch.setattr(af, "_shell_config", lambda: {"cwd": "/app", "env": {}, "timeout": 60})


def test_sandbox_harness_sets_model_timeout(monkeypatch):
    _no_harbor(monkeypatch)
    kwargs = af._sandbox_kwargs("http://s", "k", {"temperature": 1.0}, "openai/model", None)
    model_kwargs = kwargs["config"]["model"]["model_kwargs"]
    assert model_kwargs["timeout"] == 32400
    assert model_kwargs["temperature"] == 1.0


def test_host_harness_sets_model_timeout(monkeypatch):
    _no_harbor(monkeypatch)
    kwargs = af._host_kwargs("http://s", "k", {"temperature": 1.0}, "openai/model", None)
    assert kwargs["llm_call_kwargs"] == {"temperature": 1.0, "timeout": 32400}
