"""Harbor's stock in-sandbox mini-swe-agent.

Harbor installs mini-swe-agent in each trial's sandbox (``MiniSweAgent.install``:
``apt-get`` and ``uv tool install mini-swe-agent``). When the package is already
present, this agent skips that install and only applies Harbor's two runtime
patches (output cap, step timing). Either way it then adds the ContextExhausted
agent class (context_exit.py) to the agent's environment and runs mini-swe-agent
with it, and records mini-swe-agent's exit status in the agent metadata.

The pinned Harbor runs agent setup under the task's own network policy, so on a
no-network task the install cannot reach PyPI or astral.sh. ``setup`` opens
public network for the install only and restores the task's policy before the
agent runs, as the platform's evals do (mobius-platform
programs/harbor-eval/patches/harness/setup.py). A task that is already public
is set up unchanged.
"""

import asyncio
import base64
import json
import shlex
from pathlib import Path

from harbor.agents.installed import mini_swe_agent as stock
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import NetworkMode, NetworkPolicy

from miles_ext.harbor import context_exit

_PUBLIC = NetworkPolicy(network_mode=NetworkMode.PUBLIC)

INSTALL_ATTEMPTS = 4
INSTALL_BACKOFF_SECS = 15  # waits 15, 30, 45 s between attempts

_UV_PATH = 'if [ -f "$HOME/.local/bin/env" ]; then . "$HOME/.local/bin/env"; else export PATH="$HOME/.local/bin:$PATH"; fi; '
TASK_FILE = "/tmp/mswea-task.md"
# mini-swe-agent's entry point, reading --task from TASK_FILE so the task text is not in its argv
_RUN_FROM_FILE = (
    '"$(uv tool dir)/mini-swe-agent/bin/python" -c "import sys; '
    f"sys.argv = ['mini-swe-agent', *sys.argv[1:], '--task', open('{TASK_FILE}').read()]; "
    'from minisweagent.run.mini import app; app()"'
)


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
            await self._install_with_retries(environment)
        else:
            output_cap_b64 = base64.b64encode(
                stock._OUTPUT_CAP_PATCH.format(cap_bytes=stock.MAX_COMMAND_OUTPUT_BYTES).encode()
            ).decode()
            step_timing_b64 = base64.b64encode(stock._STEP_TIMING_PATCH.encode()).decode()
            await self.exec_as_agent(
                environment,
                command=(
                    "set -euo pipefail; "
                    + _UV_PATH
                    + f'echo {output_cap_b64} | base64 -d | "$(uv tool dir)/mini-swe-agent/bin/python" - && '
                    + f'echo {step_timing_b64} | base64 -d | "$(uv tool dir)/mini-swe-agent/bin/python" -'
                ),
            )
        # the ContextExhausted agent class, as a module of the agent's own environment (selected in build_cli_flags)
        module_b64 = base64.b64encode(Path(context_exit.__file__).read_bytes()).decode()
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                + _UV_PATH
                + 'py="$(uv tool dir)/mini-swe-agent/bin/python"; '
                + """site="$("$py" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; """
                + f'echo {module_b64} | base64 -d > "$site/{context_exit.AGENT_MODULE}.py" && '
                + f'"$py" -c "import {context_exit.AGENT_MODULE}"'
            ),
        )

    async def _install_with_retries(self, environment: BaseEnvironment) -> None:
        # safe to repeat: each stock install step skips what is already in place
        for attempt in range(1, INSTALL_ATTEMPTS + 1):
            try:
                return await super().install(environment)
            except NonZeroAgentExitCodeError as e:
                if attempt == INSTALL_ATTEMPTS:
                    raise
                self.logger.warning(f"harness install attempt {attempt} failed, retrying: {str(e)[:200]}")
                await asyncio.sleep(INSTALL_BACKOFF_SECS * attempt)

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        # Harbor's run renders the prompt template before building the command; the needle below must match
        self._task_text = self.render_instruction(instruction)
        return await super().run(instruction, environment, context)

    async def exec_as_agent(self, environment: BaseEnvironment, command: str, **kwargs):
        # Stock Harbor passes the task as --task=<text>, so it sits in mini-swe-agent's argv: an agent
        # command like `pkill -f /reference/program` then matches and kills the agent itself. The
        # task goes to a file instead.
        task = getattr(self, "_task_text", None)
        needle = f"mini-swe-agent --yolo --model={self.model_name} --task={shlex.quote(task)} " if task else None
        if needle and needle in command:
            task_b64 = base64.b64encode(task.encode()).decode()
            await super().exec_as_agent(environment, command=f"echo {task_b64} | base64 -d > {TASK_FILE}")
            command = command.replace(needle, f"{_RUN_FROM_FILE} --yolo --model={self.model_name} ")
        elif "mini-swe-agent --yolo" in command:
            self.logger.warning("mini-swe-agent command not recognised; the task stays in its argv (see ext-009 page)")
        return await super().exec_as_agent(environment, command=command, **kwargs)

    def build_cli_flags(self) -> str:
        return " ".join(f for f in (super().build_cli_flags(), f"--agent-class={context_exit.AGENT_CLASS}") if f)

    def populate_context_post_run(self, context: AgentContext) -> None:
        super().populate_context_post_run(context)
        # mini-swe-agent's own exit status, as the host harness records it (agent_function reads it)
        path = self.logs_dir / "mini-swe-agent.trajectory.json"
        try:
            exit_status = (json.loads(path.read_text()).get("info") or {}).get("exit_status")
        except (OSError, ValueError):
            return
        if exit_status:
            context.metadata = {**(context.metadata or {}), "mswea_exit_status": exit_status}
