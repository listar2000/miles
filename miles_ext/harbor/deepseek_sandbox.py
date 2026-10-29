"""`taskset.harness: deepseek-sandbox`: DeepSeek's own agent (dsh) inside the task sandbox (ext-046, see its patch page)."""

import asyncio
import json
import shlex
import tempfile
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent, NonZeroAgentExitCodeError
from harbor.agents.installed.node_install import nvm_node_install_snippet
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

from miles_ext.harbor import context_exit, sandbox_agent
from miles_ext.recipes.recipe import DEEPSEEK_RETAIN_RATIO

NAME = "deepseek-sandbox"
DSH_VERSION = "0.2.1-rc.3"  # the release verifiers' DeepSeek harness pins
PROFILE = "sdk-minimal"
NODE_MIN = (22, 19)  # dsh's engines floor
DIR = "$HOME/.miles-deepseek"  # the agent user's: nvm and npm install as that user, as Harbor's claude-code does
BIN = f"{DIR}/node_modules/@deepseek-ai/dsh/lib/bin.js"
FILES = "/tmp/miles-deepseek"  # patch, compat shim and task, uploaded per run
EVENTS = (EnvironmentPaths.agent_dir / "deepseek-sandbox.jsonl").as_posix()  # dsh --json events
STDERR = (EnvironmentPaths.agent_dir / "deepseek-sandbox.stderr.txt").as_posix()
AGENT_EXIT_STATUS = "agent_exit_status"  # agent metadata key the agent function reads

# dsh's Messages requests without the fields Miles' adapter rejects, sent untimed: the session server answers only
# once the whole reply is generated, past fetch's 300 s headers timeout (patch page)
COMPAT = """\
import http from "node:http";
import https from "node:https";

function post(url, init) {
  return new Promise((resolve, reject) => {
    const target = new URL(url);
    const headers = Object.fromEntries(new Headers(init.headers));
    const request = (target.protocol === "https:" ? https : http).request(
      target, {method: "POST", headers, signal: init.signal}, (response) => {
        const chunks = [];
        response.on("data", (chunk) => chunks.push(chunk));
        response.on("error", reject);
        response.on("end", () => resolve(new Response(Buffer.concat(chunks), {
          status: response.statusCode,
          statusText: response.statusMessage,
          headers: Object.entries(response.headers).flatMap(([k, v]) => [v].flat().map((x) => [k, x])),
        })));
      });
    request.on("error", reject);
    request.end(init.body);
  });
}

const fetch = globalThis.fetch;
globalThis.fetch = async (input, init) => {
  const url = input instanceof Request ? input.url : String(input);
  if (url.endsWith("/messages") && typeof init?.body === "string") {
    const body = JSON.parse(init.body);
    delete body.thinking;
    delete body.output_config;
    for (const message of body.messages ?? [])
      for (const block of Array.isArray(message.content) ? message.content : [])
        if (block.type === "tool_result") delete block.is_error;
    return post(url, {...init, body: JSON.stringify(body)});
  }
  return fetch(input, init);
};
"""

_NODE_OK = (
    'node -e \'const [a,b]=process.versions.node.split(".").map(Number); '
    f"process.exit(a>{NODE_MIN[0]}||a==={NODE_MIN[0]}&&b>={NODE_MIN[1]}?0:1)' 2>/dev/null"
)
_LOAD_NVM = 'export NVM_DIR="$HOME/.nvm"; [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"; '
# root: curl for nvm (Alpine: its own Node, since nvm's builds need glibc)
ROOT_INSTALL = (
    "if command -v apk >/dev/null 2>&1; then apk add --no-cache curl bash nodejs npm;"
    " elif ! command -v curl >/dev/null 2>&1; then"
    "  if command -v apt-get >/dev/null 2>&1; then apt-get update && apt-get install -y curl;"
    "  elif command -v dnf >/dev/null 2>&1; then dnf install -y curl;"
    "  elif command -v yum >/dev/null 2>&1; then yum install -y curl; fi;"
    " fi"
)
# agent user: Node from Harbor's nvm snippet unless the image has one new enough, then dsh; skips what is in place
AGENT_INSTALL = (
    "set -euo pipefail; "
    + _LOAD_NVM
    + f"if ! {_NODE_OK}; then {nvm_node_install_snippet(NODE_MIN[0])}; fi; "
    # the marker names the version, so another pin installs afresh
    + f'if [ ! -f "{DIR}/.ready-{DSH_VERSION}" ]; then '
    # --ignore-scripts: no native builds (task images have no compilers); the sdk-minimal tree needs none
    + f'npm install --prefix "{DIR}" --no-audit --no-fund --ignore-scripts "@deepseek-ai/dsh@{DSH_VERSION}" >/dev/null'
    + f' && touch "{DIR}/.ready-{DSH_VERSION}"; fi; '
    + f'node "{BIN}" --version'
)


def compaction(context_window: int, max_output: int, compact_at: int) -> dict[str, Any]:
    """compaction-basic's config that compacts at ``compact_at`` tokens: its trigger is
    ``min(W * thresholdRatio, W - O - headroomTokens)``."""
    retained = int((context_window - max_output) * DEEPSEEK_RETAIN_RATIO)
    if not retained < compact_at <= context_window - max_output:
        raise ValueError(
            f"compact_at_tokens {compact_at} must be above the {retained} tokens compaction keeps and at most "
            f"the context window {context_window} minus the reply cap {max_output}"
        )
    return {
        "thresholdRatio": 1,
        "headroomTokens": context_window - max_output - compact_at,
        "maxTokens": max_output,  # the summary reply's cap (default: headroomTokens)
        "retainRatio": DEEPSEEK_RETAIN_RATIO,
    }


def patch(model: str, context_window: int, max_output: int, compact_at: int | None) -> list[dict[str, Any]]:
    """The ``--patch`` over the sdk-minimal tree. A row's ``config`` replaces the bundle's whole config, so each
    changed row restates the bundle's other fields."""
    return [
        # the SDK's JSON-RPC front door and DeepSeek's telemetry uploads, as verifiers disables them
        {"id": "sdk-app-startup", "disabled": True},
        {"id": "sdk-jsonrpc-server", "disabled": True},
        {"id": "session-log-deepseek", "disabled": True},
        {"id": "plugin-package-inventory-deepseek", "disabled": True},
        {
            "id": "llm-deepseek",
            "config": {
                "apiKeyEnv": "DEEPSEEK_API_KEY",
                "defaultContextWindow": context_window,
                "streamIdleTimeoutMs": 172800000,
                "maxTokens": max_output,
            },
        },
        # bash and working_directory stay; read_image goes (Miles' Messages adapter takes no images)
        {"id": "tool-fs", "config": {"read": False, "readImage": False, "write": False, "edit": False}},
        # dsh compacts by default near its window; only with compact_at_tokens (session server v2)
        {
            "id": "compaction-basic",
            "config": compaction(context_window, max_output, compact_at) if compact_at else {"auto": False},
        },
        {
            "insert": [
                {
                    "id": "agent-default-model",
                    "name": "@deepseek-ai/dsh-agent-default-model",
                    "config": {"provider": "deepseek-official", "model": model},
                },
                # dsh's one-shot runner (its `headless` profile's row): the task from stdin, run events as JSON lines
                {"id": "headless-runner", "name": "@deepseek-ai/dsh-headless", "config": {"task": "-", "json": True}},
            ]
        },
    ]


def outcome(events: str, reply_cap: int) -> str | Exception | None:
    """From the last ``turn_end``: None (completed), ContextExhausted (the window filled), or the exception to raise."""
    from harbor.llms.base import SingleTurnMaxSeqLenExceededError

    reason, last_output = None, None
    for line in events.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "status" and event.get("phase") == "step_end":
            last_output = (event.get("usage") or {}).get("outputTokens")
        if event.get("type") == "status" and event.get("phase") == "turn_end":
            reason = event.get("reason") or {}
    if reason is None:
        return NonZeroAgentExitCodeError("dsh ended without a turn_end event")
    error = reason.get("error") or {}
    if reason.get("kind") == "completed":
        return None
    if reason.get("kind") == "max-tokens":
        # a reply shorter than the cap was cut by the server to the room left in the window
        if last_output is not None and last_output < reply_cap:
            return context_exit.EXIT_STATUS
        return SingleTurnMaxSeqLenExceededError(f"dsh reply hit its max_tokens cap of {reply_cap}")
    if error.get("code") == "CONTEXT_WINDOW_EXCEEDED":
        return context_exit.EXIT_STATUS
    return NonZeroAgentExitCodeError(f"dsh turn ended {reason.get('kind')}: {error.get('message', '')}"[:4000])


class SandboxDeepSeek(sandbox_agent.PublicSetup, BaseInstalledAgent):
    def __init__(
        self,
        *args,
        model_info: dict[str, Any] | None = None,
        max_seq_len: int | None = None,
        compact_at_tokens: int | None = None,
        **kwargs,
    ):
        """``model_info``: Miles' ``max_input_tokens`` (dsh's context window) and ``max_output_tokens`` (its reply
        cap). ``max_seq_len`` is what Miles passes every harness; dsh does not use it."""
        super().__init__(*args, **kwargs)
        info = model_info or {}
        window, self._reply_cap = int(info["max_input_tokens"]), int(info["max_output_tokens"])
        # Harbor names the model openai/<name>; dsh routes it as a deepseek-official model id
        self._patch = patch(self.model_name.rpartition("/")[2], window, self._reply_cap, compact_at_tokens)

    @staticmethod
    def name() -> str:
        return NAME

    async def install(self, environment: BaseEnvironment) -> None:
        # safe to repeat: each step skips what is already in place (as SandboxMiniSweAgent's install retries)
        for attempt in range(1, sandbox_agent.INSTALL_ATTEMPTS + 1):
            try:
                await self.exec_as_root(environment, command=ROOT_INSTALL, env={"DEBIAN_FRONTEND": "noninteractive"})
                return await self.exec_as_agent(environment, command=AGENT_INSTALL)
            except NonZeroAgentExitCodeError as e:
                if attempt == sandbox_agent.INSTALL_ATTEMPTS:
                    raise
                self.logger.warning(f"harness install attempt {attempt} failed, retrying: {str(e)[:200]}")
                await asyncio.sleep(sandbox_agent.INSTALL_BACKOFF_SECS * attempt)

    async def _upload(self, environment: BaseEnvironment, name: str, text: str) -> str:
        with tempfile.NamedTemporaryFile("w") as f:
            f.write(text)
            f.flush()
            await environment.upload_file(f.name, f"{FILES}/{name}")
        await self.exec_as_root(environment, command=f"chmod a+rX {FILES} {FILES}/{name}")  # for the agent user
        return f"{FILES}/{name}"

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        await self.exec_as_root(environment, command=f"mkdir -p {FILES}")
        # JSON is YAML; the task goes on stdin, not argv, so `pkill -f <task text>` cannot match the agent
        patch_path = await self._upload(environment, "patch.json", json.dumps(self._patch))
        compat = await self._upload(environment, "compat.mjs", COMPAT)
        task = await self._upload(environment, "task.md", self.render_instruction(instruction))
        await environment.exec(
            command=_LOAD_NVM
            + f'DSH_HOME="{DIR}/home" node --import {compat} "{BIN}" --profile {PROFILE} '
            + f"--patch {patch_path} < {task} > {EVENTS} 2> {STDERR}"
        )
        events = (await environment.exec(command=f"cat {shlex.quote(EVENTS)}")).stdout or ""
        context.n_input_tokens, context.n_output_tokens = usage(events)
        result = outcome(events, self._reply_cap)
        if result == context_exit.EXIT_STATUS:
            # the agent function keeps this status and applies agent.context_exhausted_scores_zero
            context.metadata = {**(context.metadata or {}), AGENT_EXIT_STATUS: result}
        elif result is not None:
            raise result


def usage(events: str) -> tuple[int, int]:
    """Input and output tokens summed over the run's model calls (``step_end`` usage)."""
    totals = [0, 0]
    for line in events.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("phase") == "step_end" and (u := event.get("usage")):
            totals[0] += u.get("inputTokens", 0)
            totals[1] += u.get("outputTokens", 0)
    return totals[0], totals[1]
