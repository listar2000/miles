"""Miles agent function (``--custom-agent-function-path miles_ext.harbor.agent_function.run``).

One Harbor trial per rollout sample, built on Miles' in-process Harbor agent
function (``examples/experimental/harbor/harbor_agent_function.py``), which the
fork imports from the source tree instead of vendoring a copy: an upstream sync
brings its new version along, and an API change fails loudly here. This module
adds two mini-swe-agent harnesses and owns the trial config; the upstream file
is not edited.

The harness comes from ``MILES_EXT_HARNESS`` (set by the launcher from the
recipe), falling back to the row's ``metadata.agent_name``:

mini-swe-agent-sandbox
    Harbor's stock mini-swe-agent runs *inside* the task sandbox
    (``sandbox_agent.SandboxMiniSweAgent``; task images have it preinstalled).
    Its model calls reach the session server through the launcher's Modal
    tunnel: ``MILES_SESSION_PUBLIC_URL`` is ``<tunnel-url>/<token>`` and each
    session URL ``http://<head-ip>:<port>/...`` becomes ``<public>/<port>/...``
    (``session_proxy.py`` maps it back). The tunnel host joins the agent phase's
    allowlist, so the harness reaches the model on a no-network task and the
    agent gets no other network; Harbor ignores the entry on a public task.
mini-swe-agent-host
    The mini-swe-agent loop runs in this process and executes each command in
    the sandbox (``host_agent.HostMiniSweAgent``); no tunnel needed.

Any other name goes through upstream's bindings unchanged (e.g. ``terminus-2``).

Sandbox provider: ``HARBOR_ENV_TYPE`` (a Harbor ``EnvironmentType``, read by
upstream) or ``HARBOR_ENV_IMPORT_PATH`` (an environment class Harbor loads by
import path, for providers the pinned Harbor lacks, e.g.
``miles_ext.harbor.sail:SailEnvironment``). Upstream's ``_environment_config``
only knows the first, so this module replaces it on the loaded upstream
module; with no import path set it delegates unchanged.
Both mini-swe-agent harnesses use the same settings: mini.yaml templates,
swebench.yaml shell env (cwd ``MSWEA_CWD`` or /testbed), ``HARBOR_AGENT_MAX_ITERATIONS``
steps, ``MSWEA_EXEC_TIMEOUT`` seconds per command, sampling params from the rollout.
Harbor installs the harness in each trial's sandbox; ``HARBOR_AGENT_SETUP_TIMEOUT_SEC``
(the recipe's ``agent.setup_timeout``) caps that install instead of Harbor's 360 s.
"""

import asyncio
import json
import logging
import os
import re
from importlib.metadata import version
from typing import Any
from urllib.parse import urlsplit

from miles.rollout.agentic.session import openai_session_url

from miles_ext.harbor import context_exit, upstream
from miles_ext.recipes import recipe as R

hf = upstream.harbor_agent_function()

logger = logging.getLogger(__name__)

SANDBOX_HARNESS, HOST_HARNESS = R.HARNESS_SANDBOX, R.HARNESS_HOST
# Harbor resolves agents it does not know by import path
_IMPORT_PATHS = {
    SANDBOX_HARNESS: "miles_ext.harbor.sandbox_agent:SandboxMiniSweAgent",
    HOST_HARNESS: "miles_ext.harbor.host_agent:HostMiniSweAgent",
}


def _shell_config() -> dict[str, Any]:
    from miles_ext.harbor.host_agent import load_builtin_config

    env_cfg = load_builtin_config("benchmarks/swebench.yaml").get("environment", {})
    return {
        # SWE-Gym images work in /testbed; Terminal-Bench tasks in /app
        "cwd": os.getenv("MSWEA_CWD") or env_cfg.get("cwd") or "/testbed",
        "env": dict(env_cfg.get("env") or {}),
        "timeout": int(os.getenv("MSWEA_EXEC_TIMEOUT") or env_cfg.get("timeout", 60)),
    }


def _model_call_kwargs(sampling_params: dict[str, Any]) -> dict[str, Any]:
    """litellm kwargs for one model call. Without ``timeout`` the OpenAI client's default
    (600 s) cuts a long-thinking turn, and mini-swe-agent retries it from scratch; the
    trial cap is the longest one call can usefully take."""
    return {**sampling_params, "timeout": hf._trial_timeout_s()}


def _sandbox_kwargs(session_url, api_key, sampling_params, model, max_seq_len) -> dict[str, Any]:
    """AgentConfig.kwargs for Harbor's MiniSweAgent: a mini-swe-agent config override, and the
    version the sandbox installs pinned to the one in this image (unpinned, Harbor installs the latest)."""
    model_kwargs = {**_model_call_kwargs(sampling_params), "drop_params": True, "parallel_tool_calls": False}
    return {
        "version": version("mini-swe-agent"),
        "config": {
            "agent": {"step_limit": int(os.getenv("HARBOR_AGENT_MAX_ITERATIONS", "50")), "cost_limit": 0},
            "environment": _shell_config(),
            "model": {"model_kwargs": model_kwargs},
        }
    }


def _host_kwargs(session_url, api_key, sampling_params, model, max_seq_len) -> dict[str, Any]:
    shell = _shell_config()
    return {
        "api_base": session_url,
        "api_key": api_key,
        "llm_call_kwargs": _model_call_kwargs(sampling_params),
        "step_limit": int(os.getenv("HARBOR_AGENT_MAX_ITERATIONS", "50")),
        "exec_timeout": shell["timeout"],
    }


hf.HARNESS_BINDINGS[SANDBOX_HARNESS] = hf.HarnessBinding(kwargs=_sandbox_kwargs, env=hf._mini_swe_agent_env)
hf.HARNESS_BINDINGS[HOST_HARNESS] = hf.HarnessBinding(kwargs=_host_kwargs, env=lambda url, key: {})

_upstream_environment_config = hf._environment_config
# what upstream's _environment_config reads that an import-path provider does not honor here
_TYPE_ONLY_VARS = ("HARBOR_OVERRIDE_CPUS", "HARBOR_OVERRIDE_MEMORY_MB", "HARBOR_OVERRIDE_STORAGE_MB",
                   "HARBOR_CPU_ENFORCEMENT_POLICY", "HARBOR_MEMORY_ENFORCEMENT_POLICY")


def _environment_config():
    """Upstream's environment config, or one for ``HARBOR_ENV_IMPORT_PATH`` (an environment Harbor loads by path)."""
    import_path = os.getenv("HARBOR_ENV_IMPORT_PATH", "").strip()
    if not import_path:
        return _upstream_environment_config()
    from harbor.models.trial.config import EnvironmentConfig

    if set_vars := [v for v in _TYPE_ONLY_VARS if os.getenv(v, "").strip()]:
        raise ValueError(f"{set_vars} are not applied with HARBOR_ENV_IMPORT_PATH; unset them")
    kwargs = json.loads(os.getenv("HARBOR_ENV_KWARGS", "{}") or "{}")
    return EnvironmentConfig(import_path=import_path, delete=True, kwargs=kwargs)


# build_trial_config looks the function up in its module's globals at call time
hf._environment_config = _environment_config

_SESSION_URL = re.compile(r"^http://[^/:]+:(\d+)(/.*)$")


def public_session_url(session_url: str) -> str:
    public = os.environ.get("MILES_SESSION_PUBLIC_URL")
    if not public:
        raise RuntimeError(f"{SANDBOX_HARNESS} needs the session tunnel (MILES_SESSION_PUBLIC_URL)")
    match = _SESSION_URL.match(session_url)
    if not match:
        raise RuntimeError(f"unexpected session URL {session_url!r}")
    port, path = match.groups()
    return f"{public.rstrip('/')}/{port}{path}"


def harness_for(metadata: dict[str, Any]) -> str:
    return os.getenv("MILES_EXT_HARNESS") or str(metadata.get("agent_name") or SANDBOX_HARNESS)


def build_trial_config(metadata: dict[str, Any], session_url: str, request_kwargs: dict[str, Any]):
    harness = harness_for(metadata)
    if harness == SANDBOX_HARNESS:
        session_url = public_session_url(session_url)
    config = hf.build_trial_config({**metadata, "agent_name": harness}, session_url, request_kwargs)
    if harness == SANDBOX_HARNESS:
        config.agent.extra_allowed_hosts = [*config.agent.extra_allowed_hosts, urlsplit(session_url).hostname]
        # ContextExitAgent's window (context_exit.py)
        if max_seq_len := config.agent.kwargs.get("max_seq_len"):
            config.agent.env = {**config.agent.env, context_exit.ENV_TOKENS: str(max_seq_len)}
    if import_path := _IMPORT_PATHS.get(harness):
        config.agent.name = None
        config.agent.import_path = import_path
    # upstream's agent function leaves Harbor's agent-install cap at 360 s x HARBOR_TIMEOUT_MULTIPLIER
    if setup_timeout := os.getenv("HARBOR_AGENT_SETUP_TIMEOUT_SEC"):
        config.agent.override_setup_timeout_sec = float(setup_timeout)
    return config


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Run one Harbor trial for the sample; return its verdict as metadata (same contract as upstream's run)."""
    from harbor.trial.trial import Trial

    metadata = metadata or {}
    instance_id = metadata.get("instance_id")
    trial_timeout_s = hf._trial_timeout_s()

    # config errors fail every sample alike, so they raise instead of scoring 0
    hf._ensure_provider_key()
    config = build_trial_config(metadata, openai_session_url(base_url), request_kwargs or {})

    try:
        trial = await Trial.create(config)
        result = await asyncio.wait_for(trial.run(), timeout=trial_timeout_s)
    except asyncio.TimeoutError:
        logger.error(f"Harbor trial for {instance_id} exceeded {trial_timeout_s}s; scoring 0")
        return hf._failed("TimeLimitExceeded")
    except Exception as e:
        logger.error(f"Harbor trial for {instance_id} failed: {e}", exc_info=True)
        return hf._failed("AgentError")

    out = hf.trial_result_to_metadata(result)
    # Harbor says "Submitted" whenever the verifier ran; keep the agent's ContextExhausted so filters can tell it apart
    if out["exit_status"] == "Submitted" and out["agent_metrics"].get("mswea_exit_status") == context_exit.EXIT_STATUS:
        out["exit_status"] = context_exit.EXIT_STATUS
    out["trial_dir"] = str(trial.paths.trial_dir)
    logger.info(f"Harbor trial {instance_id}: exit_status={out['exit_status']} reward={out['reward']}")
    return out
