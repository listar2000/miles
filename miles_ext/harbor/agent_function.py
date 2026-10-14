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
    (``session_proxy.py`` maps it back).
mini-swe-agent-host
    The mini-swe-agent loop runs in this process and executes each command in
    the sandbox (``host_agent.HostMiniSweAgent``); no tunnel needed.

Any other name goes through upstream's bindings unchanged (e.g. ``terminus-2``).
Both mini-swe-agent harnesses use the same settings: mini.yaml templates,
swebench.yaml shell env (cwd ``MSWEA_CWD`` or /testbed), ``HARBOR_AGENT_MAX_ITERATIONS``
steps, ``MSWEA_EXEC_TIMEOUT`` seconds per command, sampling params from the rollout.
"""

import asyncio
import logging
import os
import re
from typing import Any

from miles.rollout.agentic.session import openai_session_url

from miles_ext.harbor import upstream

hf = upstream.harbor_agent_function()

logger = logging.getLogger(__name__)

SANDBOX_HARNESS = "mini-swe-agent-sandbox"
HOST_HARNESS = "mini-swe-agent-host"
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


def _sandbox_kwargs(session_url, api_key, sampling_params, model, max_seq_len) -> dict[str, Any]:
    """AgentConfig.kwargs for Harbor's MiniSweAgent: a mini-swe-agent config override."""
    model_kwargs = {**sampling_params, "drop_params": True, "parallel_tool_calls": False}
    return {
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
        "llm_call_kwargs": dict(sampling_params),
        "step_limit": int(os.getenv("HARBOR_AGENT_MAX_ITERATIONS", "50")),
        "exec_timeout": shell["timeout"],
    }


hf.HARNESS_BINDINGS[SANDBOX_HARNESS] = hf.HarnessBinding(kwargs=_sandbox_kwargs, env=hf._mini_swe_agent_env)
hf.HARNESS_BINDINGS[HOST_HARNESS] = hf.HarnessBinding(kwargs=_host_kwargs, env=lambda url, key: {})

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
    if import_path := _IMPORT_PATHS.get(harness):
        config.agent.name = None
        config.agent.import_path = import_path
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
    out["trial_dir"] = str(trial.paths.trial_dir)
    logger.info(f"Harbor trial {instance_id}: exit_status={out['exit_status']} reward={out['reward']}")
    return out
