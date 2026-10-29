"""ext-041: Harbor's Modal sandbox for a no-network task is created filterable and narrowed before use."""

import asyncio

import pytest
from harbor.environments.modal import ModalEnvironment as Upstream
from harbor.models.task.config import NetworkMode, NetworkPolicy

from miles_ext.sandboxes import modal_env

NONE = NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
PUBLIC = NetworkPolicy(network_mode=NetworkMode.PUBLIC)


class _Call:
    def __init__(self, log, name, fail=False):
        self.log, self.name, self.fail = log, name, fail

    async def aio(self, **kwargs):
        self.log.append((self.name, kwargs))
        if self.fail:
            raise RuntimeError(self.name)


class _Sandbox:
    def __init__(self, log, fail_set=False):
        self._experimental_set_outbound_network_policy = _Call(log, "set", fail_set)
        self.terminate = _Call(log, "terminate")


def _env(monkeypatch, policy, dynamic, fail_set=False):
    log = []

    async def create(self, **kwargs):  # upstream builds the filter from the policy current at create
        log.append(("create", Upstream._dynamic_network_kwargs(self.network_policy)))
        return _Sandbox(log, fail_set)

    monkeypatch.setattr(Upstream, "_create_sandbox", create)
    env = object.__new__(modal_env.ModalEnvironment)
    env._network_policy, env._dynamic_network = policy, dynamic
    return env, log


def test_a_no_network_task_with_phases_starts_filterable_and_closed(monkeypatch):
    env, log = _env(monkeypatch, NONE, dynamic=True)
    asyncio.run(env._create_sandbox(entrypoint=["sleep"]))
    assert log == [
        ("create", {"outbound_domain_allowlist": ["*"], "outbound_cidr_allowlist": ["0.0.0.0/0"]}),
        ("set", {"outbound_domain_allowlist": [], "outbound_cidr_allowlist": []}),
    ]
    assert env.network_policy == NONE


@pytest.mark.parametrize("policy,dynamic", [(PUBLIC, True), (NONE, False)])
def test_every_other_sandbox_is_created_as_upstream_does(monkeypatch, policy, dynamic):
    env, log = _env(monkeypatch, policy, dynamic)
    asyncio.run(env._create_sandbox())
    assert [name for name, _ in log] == ["create"]


def test_a_sandbox_that_cannot_be_closed_is_terminated(monkeypatch):
    env, log = _env(monkeypatch, NONE, dynamic=True, fail_set=True)
    with pytest.raises(RuntimeError, match="set"):
        asyncio.run(env._create_sandbox())
    assert [name for name, _ in log] == ["create", "set", "terminate"]
    assert env.network_policy == NONE


def test_harbors_modal_type_loads_this_class_and_keeps_upstreams_fields(monkeypatch):
    from miles_ext.harbor import agent_function as af

    monkeypatch.delenv("HARBOR_ENV_IMPORT_PATH", raising=False)
    monkeypatch.setenv("HARBOR_ENV_TYPE", "modal")
    monkeypatch.setenv("HARBOR_ENV_KWARGS", '{"app_name": "miles-mobius-x"}')
    monkeypatch.setenv("HARBOR_CPU_ENFORCEMENT_POLICY", "limit")
    upstream = af._upstream_environment_config()
    config = af._environment_config()
    assert config.import_path == modal_env.IMPORT_PATH and config.type is None
    assert config.model_dump(exclude={"type", "import_path"}) == upstream.model_dump(exclude={"type", "import_path"})
