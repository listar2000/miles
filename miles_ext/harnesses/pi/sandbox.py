"""`taskset.harness: pi-sandbox`: Pi (pi-coding-agent) inside the task sandbox (ext-055, see its patch page).

Pi runs unpatched, configured as the Mobius platform runs it for evals (``MobiusPi`` in mobius-platform
``programs/harbor-eval/patches/harness/pi.py`` on ``staging``): a plain tool-call loop that loads nothing a task,
image or home directory could add, offers Pi's seven built-in tools and never compacts. The model is Miles'
session server, as a custom OpenAI-completions provider in Pi's ``models.json``.
"""

import json
import shlex
import tempfile
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent, NonZeroAgentExitCodeError
from harbor.agents.installed.node_install import nvm_node_install_snippet
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

from miles_ext.harnesses import common
from miles_ext.harnesses.mini_swe_agent import context_exit

NAME = "pi-sandbox"
PI_PACKAGE = "@earendil-works/pi-coding-agent"
PI_VERSION = "1.5.0"  # the release Mobius pins (MobiusPi.PI_VERSION)
# Node the GLM-5.3-Flash traces ran on: Pi's system prompt names its install path, which holds this version
NODE_VERSION = "22.23.4"
CONFIG_DIR = "/tmp/miles-pi"  # PI_CODING_AGENT_DIR: settings.json and models.json, uploaded per run
PROVIDER = "miles"
API_KEY_ENV = "MILES_PI_API_KEY"
TOOLS = "read,write,edit,bash,grep,find,ls"
# nothing from the task, image or home: discovered extensions, skills, prompt templates, AGENTS.md, MCP
PLAIN_FLAGS = "--no-extensions --no-skills --no-prompt-templates --no-context-files --no-mcp --no-approve"
# compaction off stops both threshold and overflow compaction
SETTINGS = {"compaction": {"enabled": False}}
SESSIONS = (EnvironmentPaths.agent_dir / "pi" / "sessions").as_posix()
EVENTS = (EnvironmentPaths.agent_dir / "pi.jsonl").as_posix()  # --mode json events, without message_update
STDERR = (EnvironmentPaths.agent_dir / "pi.stderr.txt").as_posix()
AGENT_EXIT_STATUS = common.AGENT_EXIT_STATUS
# what an over-long request's 400 says (SGLang's context-length check)
_CONTEXT_ERRORS = ("context length", "context window", "maximum context")

# agent user: Node and Pi as Harbor's Pi installs them (global npm under nvm); skips what is in place
AGENT_INSTALL = (
    "set -euo pipefail; "
    + common.LOAD_NVM
    + f'if ! command -v pi >/dev/null 2>&1 || [ "$(pi --version 2>/dev/null | tail -n1)" != "{PI_VERSION}" ]; then '
    + f"{nvm_node_install_snippet(NODE_VERSION)} && "
    # --ignore-scripts: no native builds (task images have no compilers)
    + f"npm install -g --no-audit --no-fund --ignore-scripts {PI_PACKAGE}@{PI_VERSION} >/dev/null; fi; "
    + "pi --version"
)


def models_json(base_url: str, model: str, context_window: int, max_output: int) -> dict[str, Any]:
    """Pi's provider for Miles' session server. Pi's own compat defaults for an OpenAI-style server: no strict
    tool schemas and no thinking parameters, so the chat template decides reasoning."""
    return {
        "providers": {
            PROVIDER: {
                "baseUrl": base_url,
                "apiKey": API_KEY_ENV,
                "api": "openai-completions",
                "models": [{"id": model, "contextWindow": context_window, "maxTokens": max_output}],
            }
        }
    }


def command(model: str, instruction: str) -> str:
    return (
        common.LOAD_NVM
        + f"pi --print --mode json --session-dir {SESSIONS} --provider {PROVIDER} --model {shlex.quote(model)} "
        + f"{PLAIN_FLAGS} --tools {TOOLS} {shlex.quote(instruction)} "
        # message_update events repeat the growing reply; Harbor's Pi drops them too
        + f'2> {STDERR} </dev/null | grep -v \'"type":"message_update"\' > {EVENTS}'
    )


def _assistant_ends(events: str):
    for line in events.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        message = event.get("message") or {}
        if event.get("type") == "message_end" and message.get("role") == "assistant":
            yield message


def outcome(events: str, reply_cap: int) -> str | Exception | None:
    """From Pi's last reply: None (completed), ContextExhausted (the window filled), or the exception to raise."""
    from harbor.llms.base import SingleTurnMaxSeqLenExceededError

    last = None
    for last in _assistant_ends(events):
        pass
    if last is None:
        return NonZeroAgentExitCodeError("pi ended without a reply")
    reason = last.get("stopReason")
    if reason == "stop":
        return None
    if reason == "length":
        # a reply shorter than the cap was cut by the server to the room left in the window
        if (last.get("usage") or {}).get("output", reply_cap) < reply_cap:
            return context_exit.EXIT_STATUS
        return SingleTurnMaxSeqLenExceededError(f"pi reply hit its max_tokens cap of {reply_cap}")
    error = str(last.get("errorMessage") or "")
    if reason == "error" and any(marker in error.lower() for marker in _CONTEXT_ERRORS):
        return context_exit.EXIT_STATUS
    return NonZeroAgentExitCodeError(f"pi stopped with {reason}: {error}"[:4000])


def usage(events: str) -> tuple[int, int]:
    """Input and output tokens summed over the run's model calls."""
    totals = [0, 0]
    for message in _assistant_ends(events):
        u = message.get("usage") or {}
        totals[0] += u.get("input", 0) + u.get("cacheRead", 0)
        totals[1] += u.get("output", 0)
    return totals[0], totals[1]


class SandboxPi(common.PublicSetup, BaseInstalledAgent):
    def __init__(
        self,
        *args,
        base_url: str,
        model_info: dict[str, Any] | None = None,
        max_seq_len: int | None = None,
        **kwargs,
    ):
        """``base_url``: the session's OpenAI root (``.../v1``). ``model_info``: Miles' ``max_input_tokens`` (Pi's
        context window) and ``max_output_tokens`` (its reply cap). ``max_seq_len`` is what Miles passes every
        harness; Pi does not use it."""
        super().__init__(*args, **kwargs)
        info = model_info or {}
        window, self._reply_cap = int(info["max_input_tokens"]), int(info["max_output_tokens"])
        # Harbor names the model openai/<name>; the session server takes the bare name
        self._model = self.model_name.rpartition("/")[2]
        self._models = models_json(base_url, self._model, window, self._reply_cap)

    @staticmethod
    def name() -> str:
        return NAME

    async def install(self, environment: BaseEnvironment) -> None:
        await common.install(self, environment, AGENT_INSTALL)

    async def _upload(self, environment: BaseEnvironment, name: str, text: str) -> None:
        with tempfile.NamedTemporaryFile("w") as f:
            f.write(text)
            f.flush()
            await environment.upload_file(f.name, f"{CONFIG_DIR}/{name}")

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        await self.exec_as_root(environment, command=f"mkdir -p {CONFIG_DIR}")
        await self._upload(environment, "settings.json", json.dumps(SETTINGS) + "\n")
        await self._upload(environment, "models.json", json.dumps(self._models, indent=2) + "\n")
        await self.exec_as_root(environment, command=f"chmod -R a+rwX {CONFIG_DIR}")  # Pi writes auth.json there
        await environment.exec(
            command=command(self._model, self.render_instruction(instruction)),
            env={
                "PI_CODING_AGENT_DIR": CONFIG_DIR,
                "PI_OFFLINE": "1",
                "PI_TELEMETRY": "0",
                API_KEY_ENV: "dummy",
            },
        )
        events = (await environment.exec(command=f"cat {shlex.quote(EVENTS)}")).stdout or ""
        context.n_input_tokens, context.n_output_tokens = usage(events)
        result = outcome(events, self._reply_cap)
        if result == context_exit.EXIT_STATUS:
            # the agent function keeps this status and applies agent.context_exhausted_scores_zero
            context.metadata = {**(context.metadata or {}), AGENT_EXIT_STATUS: result}
        elif result is not None:
            raise result
