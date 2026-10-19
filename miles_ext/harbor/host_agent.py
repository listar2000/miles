"""mini-swe-agent run host-side as a Harbor agent.

Harbor's stock ``mini-swe-agent`` installs and runs the agent *inside* the task
sandbox, so its model calls must route from the sandbox back to the trainer's
session server. On Modal that route does not exist in the shape Miles expects.
This agent runs mini-swe-agent's ``DefaultAgent`` loop in the rollout worker
instead: model calls go to the per-sample session URL (reachable locally), and
every bash action is executed in the sandbox through Harbor's
``environment.exec``. Harbor's trial lifecycle and verifier are unchanged.

Token parity is unaffected: every model call still goes through Miles' session
server, which records the exact sampled tokens.

It matches the in-sandbox harness (``sandbox_agent.py``) where that changes the
trajectory or its metrics: the context-window exit (``context_exit.ContextExit``),
Harbor's per-command output cap, and Harbor's per-step LLM-wait and command
timings in the agent metadata.

kwargs (via ``AgentConfig.kwargs``):
  api_base         per-sample OpenAI-compatible session URL (required)
  api_key          any non-empty string
  llm_call_kwargs  sampling params forwarded to every completion call
  step_limit       max model calls (default 30)
  config_name      mini-swe-agent builtin config for the agent and model
                   templates (default ``mini.yaml``, whose submit step is a
                   bare ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``)
  shell            ``{"cwd", "env", "timeout"}`` for each command; cwd "" runs
                   where the sandbox starts (default: ``benchmarks/swebench.yaml``,
                   cwd /testbed, 60 s per command)
  max_seq_len      context window; the agent ends "ContextExhausted" at it
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml
from harbor.agents.base import BaseAgent
from harbor.agents.installed import mini_swe_agent as stock
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from miles_ext.harbor import context_exit

logger = logging.getLogger(__name__)

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

# One thread per running episode. Under --custom-agent-function-mode inline all
# episodes share the rollout process, and asyncio's default executor (at most
# 32 threads) would serialize them.
_AGENT_THREADS = ThreadPoolExecutor(max_workers=512, thread_name_prefix="mswea")


def load_builtin_config(name: str) -> dict[str, Any]:
    from minisweagent.config import builtin_config_dir

    return yaml.safe_load((Path(builtin_config_dir) / name).read_text())


class _HarborExecEnvironment:
    """mini-swe-agent ``Environment`` that runs each action in a Harbor sandbox.

    ``execute`` is called from the agent's worker thread and hops onto the
    trial's event loop, where the Harbor environment lives.
    """

    def __init__(self, environment: BaseEnvironment, loop: asyncio.AbstractEventLoop, *, cwd: str | None,
                 env: dict[str, str], timeout: int):
        self._environment = environment
        self._loop = loop
        self.config = {"cwd": cwd, "env": env, "timeout": timeout}
        self.uname = {"system": "Linux", "release": "", "version": "", "machine": "x86_64"}

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        from minisweagent.exceptions import Submitted

        command = action.get("command", "")
        timeout = timeout or self.config["timeout"]
        future = asyncio.run_coroutine_threadsafe(
            self._environment.exec(
                command, cwd=cwd or self.config["cwd"] or None, env=self.config["env"], timeout_sec=timeout
            ),
            self._loop,
        )
        try:
            result = future.result(timeout=timeout + 120)
            text = (result.stdout or "") + (result.stderr or "")
            output = {"output": text, "returncode": result.return_code, "exception_info": ""}
        except Exception as e:  # timeouts and sandbox errors become observations
            output = {
                "output": "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {type(e).__name__}: {e}",
            }
        output["output"] = cap_output(output["output"])
        lines = output["output"].lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == SUBMIT_MARKER and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {"role": "exit", "content": submission, "extra": {"exit_status": "Submitted", "submission": submission}}
            )
        return output

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        # the instance template prints uname fields (the in-sandbox agent reads its own)
        return {**self.config, **self.uname, **kwargs}

    def serialize(self) -> dict:
        return {"info": {"config": {"environment": self.config, "environment_type": "harbor-exec"}}}


def cap_output(text: str, cap: int = stock.MAX_COMMAND_OUTPUT_BYTES) -> str:
    """Harbor's in-sandbox output cap (``stock._OUTPUT_CAP_PATCH``): keep head and tail."""
    if len(text) <= cap:
        return text
    return text[: cap // 2] + "\n...[harbor: elided " + str(len(text) - cap) + " chars]...\n" + text[-(cap // 2) :]


def _host_agent_class():
    from minisweagent.agents.default import DefaultAgent

    class _HostAgent(context_exit.ContextExit, DefaultAgent):
        """DefaultAgent with the window exit and the timings Harbor's step-timing patch stamps in the sandbox."""

        def __init__(self, *args, context_limit: int = 0, **kwargs):
            super().__init__(*args, **kwargs)
            self._context_limit = context_limit

        def context_limits(self) -> tuple[int, int]:
            return self._context_limit, super().context_limits()[1]

        def query(self) -> dict:
            t0 = time.time()
            message = super().query()
            message.setdefault("extra", {})["harbor_llm_wait_msec"] = (time.time() - t0) * 1000
            return message

        def execute_actions(self, message: dict) -> list[dict]:
            t0 = time.time()
            observations = super().execute_actions(message)
            if observations:
                observations[0].setdefault("extra", {})["harbor_tool_exec_msec"] = (time.time() - t0) * 1000
            return observations

    return _HostAgent


class HostMiniSweAgent(BaseAgent):
    SUPPORTS_ATIF = False

    def __init__(
        self,
        *args,
        api_base: str,
        api_key: str = "dummy",
        llm_call_kwargs: dict[str, Any] | None = None,
        step_limit: int = 30,
        config_name: str = "mini.yaml",
        shell: dict[str, Any] | None = None,
        max_seq_len: int | None = None,
        **kwargs,
    ):
        # Miles' agent function passes model_info for other harnesses
        kwargs.pop("model_info", None)
        super().__init__(*args, **kwargs)
        self._api_base = api_base
        self._api_key = api_key
        self._llm_call_kwargs = dict(llm_call_kwargs or {})
        self._step_limit = step_limit
        self._config = load_builtin_config(config_name)
        self._shell = shell or load_builtin_config("benchmarks/swebench.yaml").get("environment", {})
        self._max_seq_len = int(max_seq_len or 0)

    @staticmethod
    def name() -> str:
        return "mini-swe-agent-host"

    def version(self) -> str | None:
        import minisweagent

        return minisweagent.__version__

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    def _build_agent(self, environment: BaseEnvironment, loop: asyncio.AbstractEventLoop):
        from minisweagent.models.litellm_model import LitellmModel

        model_cfg = dict(self._config.get("model", {}))
        model_kwargs = {
            **(model_cfg.get("model_kwargs") or {}),
            **self._llm_call_kwargs,
            "api_base": self._api_base,
            "api_key": self._api_key,
            "drop_params": True,
            # one command per turn keeps the trajectory a plain linear chat
            "parallel_tool_calls": False,
        }
        model = LitellmModel(
            model_name=self.model_name or "openai/model",
            model_kwargs=model_kwargs,
            cost_tracking="ignore_errors",
            observation_template=model_cfg["observation_template"],
            format_error_template=model_cfg["format_error_template"],
        )
        env = _HarborExecEnvironment(
            environment,
            loop,
            cwd=self._shell.get("cwd"),
            env=dict(self._shell.get("env") or {}),
            timeout=int(self._shell.get("timeout") or 60),
        )
        agent_cfg = {
            k: v for k, v in self._config.get("agent", {}).items() if k in ("system_template", "instance_template")
        }
        return _host_agent_class()(
            model,
            env,
            **agent_cfg,
            step_limit=self._step_limit,
            cost_limit=0,
            output_path=self.logs_dir / "mini-swe-agent.trajectory.json",
            context_limit=self._max_seq_len,
        )

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        loop = asyncio.get_running_loop()
        agent = self._build_agent(environment, loop)
        try:
            result = await environment.exec("uname -s; uname -r; uname -v; uname -m", timeout_sec=60)
            agent.env.uname.update(zip(("system", "release", "version", "machine"), (result.stdout or "").splitlines()))
        except Exception as e:
            logger.warning(f"uname in the sandbox failed, the prompt reports Linux x86_64: {e}")
        exit_status = "unknown"
        try:
            result = await loop.run_in_executor(_AGENT_THREADS, agent.run, instruction)
            exit_status = result.get("exit_status", "unknown")
        except Exception as e:
            # A failed model call (e.g. context overflow) ends the episode; the
            # verifier still grades whatever state the repo is in.
            exit_status = type(e).__name__
            logger.warning(f"mini-swe-agent ended with {exit_status}: {e}")
        finally:
            context.n_steps = agent.n_calls
            context.metadata = {
                **(context.metadata or {}),
                **stock._extract_subtiming_msec({"messages": agent.messages}),
                "mswea_exit_status": exit_status,
                "turns": agent.n_calls,
            }
