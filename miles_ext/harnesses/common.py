"""What the in-sandbox harnesses share: setup with public network, and the install retry budget."""

from harbor.environments.base import BaseEnvironment
from harbor.models.task.config import NetworkMode, NetworkPolicy

_PUBLIC = NetworkPolicy(network_mode=NetworkMode.PUBLIC)

INSTALL_ATTEMPTS = 4
INSTALL_BACKOFF_SECS = 15  # waits 15, 30, 45 s between attempts


class PublicSetup:
    """Run agent setup with public network and restore the task's policy before returning."""

    async def setup(self, environment: BaseEnvironment) -> None:
        baseline = environment.network_policy
        if baseline == _PUBLIC:
            return await super().setup(environment)
        await environment.set_network_policy(_PUBLIC)
        try:
            await super().setup(environment)
        finally:
            # failure to restore must abort the trial, never start a solver online
            await environment.set_network_policy(baseline)
