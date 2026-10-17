"""Harbor's stock in-sandbox mini-swe-agent.

Harbor installs mini-swe-agent in each trial's sandbox (``MiniSweAgent.install``:
``apt-get`` and ``uv tool install mini-swe-agent``). When the package is already
present, this agent skips that install and only applies Harbor's two runtime
patches (output cap, step timing).

The pinned Harbor runs agent setup under the task's own network policy, so on a
no-network task the install cannot reach PyPI or astral.sh. ``setup`` opens
public network for the install only and restores the task's policy before the
agent runs, as the platform's evals do (mobius-platform
programs/harbor-eval/patches/harness/setup.py). A task that is already public
is set up unchanged.
"""

import base64

from harbor.agents.installed import mini_swe_agent as stock
from harbor.environments.base import BaseEnvironment
from harbor.models.task.config import NetworkMode, NetworkPolicy

from miles_ext.harbor import context_exit

_PUBLIC = NetworkPolicy(network_mode=NetworkMode.PUBLIC)

_UV_PATH = 'if [ -f "$HOME/.local/bin/env" ]; then . "$HOME/.local/bin/env"; else export PATH="$HOME/.local/bin:$PATH"; fi; '


class SandboxMiniSweAgent(stock.MiniSweAgent):
    @staticmethod
    def name() -> str:
        return "mini-swe-agent-sandbox"

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

    async def install(self, environment: BaseEnvironment) -> None:
        probe = await environment.exec(command=_UV_PATH + "command -v mini-swe-agent")
        if probe.return_code != 0:
            await super().install(environment)
        else:
            output_cap_b64 = base64.b64encode(
                stock._OUTPUT_CAP_PATCH.format(cap_bytes=stock.MAX_COMMAND_OUTPUT_BYTES).encode()
            ).decode()
            step_timing_b64 = base64.b64encode(stock._STEP_TIMING_PATCH.encode()).decode()
            await self._run_patches(environment, output_cap_b64, step_timing_b64)
        # after Harbor's step-timing patch, whose query line it wraps
        await self._run_patches(environment, base64.b64encode(context_exit.PATCH.encode()).decode())

    async def _run_patches(self, environment: BaseEnvironment, *scripts_b64: str) -> None:
        python = '"$(uv tool dir)/mini-swe-agent/bin/python"'
        await self.exec_as_agent(
            environment,
            command="set -euo pipefail; " + _UV_PATH
            + " && ".join(f"echo {b64} | base64 -d | {python} -" for b64 in scripts_b64),
        )
