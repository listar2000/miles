"""Harbor's Modal sandbox provider, able to start a no-network task whose policy changes during the trial (ext-041).

A no-network task's policy changes twice in a trial: ``SandboxMiniSweAgent.setup`` opens public network for
the harness install, and the agent phase allows the session tunnel's host. The pinned Harbor sees the change
and creates the sandbox with the task's own policy as its outbound filter, which for no network is an empty
domain list. Modal reads an empty list as "no filter", and a sandbox created without a filter refuses
any later filter ("sandbox was created without a domain allowlist; enabling domain filtering while running
is not currently supported"), so the first change fails the trial.

Here such a sandbox is created with the allow-all filter (``*``), which Modal can change later, and is
narrowed to the task's policy before the call returns, so nothing runs in it with network. Every other
sandbox is created exactly as upstream creates it. It replaces Harbor's class wherever a run names Harbor's
``modal`` type (``HARBOR_ENV_TYPE=modal``, what ``sandbox.provider: modal`` sets): the agent function swaps it in,
keeping every other field of upstream's config.
"""

from harbor.environments.modal import ModalEnvironment as _Upstream
from harbor.models.task.config import NetworkMode, NetworkPolicy

IMPORT_PATH = "miles_ext.sandboxes.modal_env:ModalEnvironment"
REPLACES_TYPE = "modal"  # the Harbor EnvironmentType this class stands in for

_OPEN = NetworkPolicy(network_mode=NetworkMode.PUBLIC)


class ModalEnvironment(_Upstream):
    async def _create_sandbox(self, **kwargs):
        if not (self._dynamic_network and self.network_policy.network_mode == NetworkMode.NO_NETWORK):
            return await super()._create_sandbox(**kwargs)
        baseline = self.network_policy
        self._network_policy = _OPEN  # upstream builds the create-time filter from the current policy
        try:
            sandbox = await super()._create_sandbox(**kwargs)
        finally:
            self._network_policy = baseline
        try:
            await sandbox._experimental_set_outbound_network_policy.aio(**self._dynamic_network_kwargs(baseline))
        except BaseException:
            await sandbox.terminate.aio()  # never leave an open sandbox behind for a no-network task
            raise
        return sandbox
