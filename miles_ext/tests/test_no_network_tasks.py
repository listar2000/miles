"""The in-sandbox harness works on a no-network task: its install gets public network
for setup only, and the agent phase may reach the session tunnel and nothing else:

    python -m pytest miles_ext/tests/test_no_network_tasks.py
"""

import asyncio
from types import SimpleNamespace

import pytest
from harbor.agents.installed import mini_swe_agent as stock
from harbor.models.task.config import NetworkMode, NetworkPolicy

from miles_ext.harbor import agent_function as af
from miles_ext.harbor.sandbox_agent import SandboxMiniSweAgent

NO_NETWORK = NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
PUBLIC = NetworkPolicy(network_mode=NetworkMode.PUBLIC)


class _Env:
    def __init__(self, policy):
        self.network_policy = policy
        self.switches = []

    async def set_network_policy(self, policy):
        self.switches.append(policy.network_mode)
        self.network_policy = policy


def _setup(monkeypatch, env, fail=False):
    seen = []

    async def stock_setup(self, environment):
        seen.append(environment.network_policy.network_mode)
        if fail:
            raise RuntimeError("Agent install failed")

    monkeypatch.setattr(stock.MiniSweAgent, "setup", stock_setup)
    asyncio.run(SandboxMiniSweAgent.setup(object.__new__(SandboxMiniSweAgent), env))
    return seen


def test_no_network_task_installs_with_public_network_then_restores(monkeypatch):
    env = _Env(NO_NETWORK)
    assert _setup(monkeypatch, env) == [NetworkMode.PUBLIC]
    assert env.network_policy == NO_NETWORK


def test_failed_install_still_restores_the_task_policy(monkeypatch):
    env = _Env(NO_NETWORK)
    with pytest.raises(RuntimeError):
        _setup(monkeypatch, env, fail=True)
    assert env.network_policy == NO_NETWORK


def test_public_task_is_set_up_unchanged(monkeypatch):
    env = _Env(PUBLIC)
    assert _setup(monkeypatch, env) == [NetworkMode.PUBLIC]
    assert env.switches == []


def _upstream_config(metadata, session_url, request_kwargs):
    return SimpleNamespace(
        agent=SimpleNamespace(
            name=metadata["agent_name"],
            import_path=None,
            override_setup_timeout_sec=None,
            extra_allowed_hosts=[],
            kwargs={"max_seq_len": 196608},
            env={},
        )
    )


def test_agent_phase_allows_the_session_tunnel_host(monkeypatch):
    monkeypatch.setattr(af.hf, "build_trial_config", _upstream_config)
    monkeypatch.setenv("MILES_EXT_HARNESS", af.SANDBOX_HARNESS)
    monkeypatch.setenv("MILES_SESSION_PUBLIC_URL", "https://abc123.r9.modal.host/tok")
    config = af.build_trial_config({"instance_id": "x"}, "http://10.0.0.1:30000/sessions/s/v1", {})
    assert config.agent.extra_allowed_hosts == ["abc123.r9.modal.host"]
    assert config.agent.env == {"MSWEA_CONTEXT_TOKENS": "196608"}


def test_host_harness_adds_no_allowlist(monkeypatch):
    monkeypatch.setattr(af.hf, "build_trial_config", _upstream_config)
    monkeypatch.setenv("MILES_EXT_HARNESS", af.HOST_HARNESS)
    config = af.build_trial_config({"instance_id": "x"}, "http://10.0.0.1:30000/sessions/s/v1", {})
    assert config.agent.extra_allowed_hosts == []
