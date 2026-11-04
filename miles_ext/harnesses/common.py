"""What the in-sandbox harnesses share: setup with public network, the install retry budget, and the Node bootstrap."""

import asyncio

from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.environments.base import BaseEnvironment
from harbor.models.task.config import NetworkMode, NetworkPolicy

_PUBLIC = NetworkPolicy(network_mode=NetworkMode.PUBLIC)

LOAD_NVM = 'export NVM_DIR="$HOME/.nvm"; [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"; '
# root: curl for nvm (Alpine: its own Node, since nvm's builds need glibc)
ROOT_INSTALL = (
    "if command -v apk >/dev/null 2>&1; then apk add --no-cache curl bash nodejs npm;"
    " elif ! command -v curl >/dev/null 2>&1; then"
    "  if command -v apt-get >/dev/null 2>&1; then apt-get update && apt-get install -y curl;"
    "  elif command -v dnf >/dev/null 2>&1; then dnf install -y curl;"
    "  elif command -v yum >/dev/null 2>&1; then yum install -y curl; fi;"
    " fi"
)

# agent metadata key where an in-sandbox harness reports ContextExhausted; the agent function reads it
AGENT_EXIT_STATUS = "agent_exit_status"
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


async def install(agent, environment: BaseEnvironment, agent_command: str) -> None:
    """``ROOT_INSTALL`` as root, then ``agent_command`` as the agent user, retried: each step must skip what is
    already in place (as mini-swe-agent's install retries)."""
    for attempt in range(1, INSTALL_ATTEMPTS + 1):
        try:
            await agent.exec_as_root(environment, command=ROOT_INSTALL, env={"DEBIAN_FRONTEND": "noninteractive"})
            return await agent.exec_as_agent(environment, command=agent_command)
        except NonZeroAgentExitCodeError as e:
            if attempt == INSTALL_ATTEMPTS:
                raise
            agent.logger.warning(f"harness install attempt {attempt} failed, retrying: {str(e)[:200]}")
            await asyncio.sleep(INSTALL_BACKOFF_SECS * attempt)
