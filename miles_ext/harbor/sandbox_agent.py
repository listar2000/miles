"""Harbor's stock in-sandbox mini-swe-agent, minus the per-trial install.

Harbor's ``MiniSweAgent.install`` runs ``apt-get`` and ``uv tool install
mini-swe-agent`` inside every trial's sandbox, which took ~8 minutes per
episode on a SWE-Gym image. The task images built by ``make_sandbox_tasks.py``
already carry the package, so this agent only applies Harbor's two
runtime patches (output cap, step timing) and falls back to the full install
when the package is missing.

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
            await environment.set_network_policy(baseline)

    async def install(self, environment: BaseEnvironment) -> None:
        probe = await environment.exec(command=_UV_PATH + "command -v mini-swe-agent")
        if probe.return_code != 0:
            return await super().install(environment)
        output_cap_b64 = base64.b64encode(
            stock._OUTPUT_CAP_PATCH.format(cap_bytes=stock.MAX_COMMAND_OUTPUT_BYTES).encode()
        ).decode()
        step_timing_b64 = base64.b64encode(stock._STEP_TIMING_PATCH.encode()).decode()
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; " + _UV_PATH
                + f'echo {output_cap_b64} | base64 -d | "$(uv tool dir)/mini-swe-agent/bin/python" - && '
                + f'echo {step_timing_b64} | base64 -d | "$(uv tool dir)/mini-swe-agent/bin/python" -'
            ),
        )
