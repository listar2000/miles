"""The recipe's agent.setup_timeout reaches Harbor as the agent-install cap, replacing
Harbor's 360 s (x HARBOR_TIMEOUT_MULTIPLIER, which recipes set to 1.0):

    python -m pytest miles_ext/tests/test_agent_setup_timeout.py
"""

from types import SimpleNamespace

from miles_ext.harbor import agent_function as af
from miles_ext.recipes import recipe as R


def _upstream_config(metadata, session_url, request_kwargs):
    return SimpleNamespace(agent=SimpleNamespace(name=metadata["agent_name"], import_path=None,
                                                 override_setup_timeout_sec=None),
                           verifier=SimpleNamespace(disable=False))


def test_recipe_default_reaches_harbor_env():
    recipe = R.from_dict({"name": "t", "model": {"preset": "qwen3.5-9b"}})
    assert R.harbor_env(recipe, "t-1")["HARBOR_AGENT_SETUP_TIMEOUT_SEC"] == "1800"


def test_trial_config_carries_setup_timeout(monkeypatch):
    monkeypatch.setattr(af.hf, "build_trial_config", _upstream_config)
    monkeypatch.setenv("MILES_EXT_HARNESS", af.HOST_HARNESS)
    monkeypatch.setenv("HARBOR_AGENT_SETUP_TIMEOUT_SEC", "2400")
    config = af.build_trial_config({"instance_id": "x"}, "http://s/v1", {})
    assert config.agent.override_setup_timeout_sec == 2400.0


def test_unset_keeps_harbor_default(monkeypatch):
    monkeypatch.setattr(af.hf, "build_trial_config", _upstream_config)
    monkeypatch.setenv("MILES_EXT_HARNESS", af.HOST_HARNESS)
    monkeypatch.delenv("HARBOR_AGENT_SETUP_TIMEOUT_SEC", raising=False)
    config = af.build_trial_config({"instance_id": "x"}, "http://s/v1", {})
    assert config.agent.override_setup_timeout_sec is None


def test_benchmark_trials_can_skip_the_verifier(monkeypatch):
    """ext-040's benchmark sets MILES_EXT_SANDBOX_PROFILE_NO_VERIFIER=1; training never does."""
    from miles_ext.obs import sandbox_profile

    monkeypatch.setattr(af.hf, "build_trial_config", _upstream_config)
    monkeypatch.setenv("MILES_EXT_HARNESS", af.HOST_HARNESS)
    monkeypatch.delenv(sandbox_profile.NO_VERIFIER_ENV, raising=False)
    assert af.build_trial_config({"instance_id": "x"}, "http://s/v1", {}).verifier.disable is False
    monkeypatch.setenv(sandbox_profile.NO_VERIFIER_ENV, "1")
    assert af.build_trial_config({"instance_id": "x"}, "http://s/v1", {}).verifier.disable is True
