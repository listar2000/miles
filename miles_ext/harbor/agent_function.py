"""Miles agent function (``--custom-agent-function-path miles_ext.harbor.agent_function.run``).

One Harbor trial per rollout sample, built on Miles' in-process Harbor agent
function (``examples/experimental/harbor/harbor_agent_function.py``), which the
fork imports from the source tree instead of vendoring a copy: an upstream sync
brings its new version along, and an API change fails loudly here. This module
adds three in-fork harnesses and owns the trial config; the upstream file is
not edited.

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
terminus-2-sandbox
    Harbor's Terminus-2 runs inside the task sandbox
    (``terminus_sandbox.SandboxTerminus2``, ext-045) with upstream's terminus-2
    kwargs, reaching the model through the tunnel as mini-swe-agent-sandbox
    does. ``MILES_EXT_COMPACT_AT_TOKENS`` (recipe ``agent.compact_at_tokens``)
    turns on its summarization at that many context tokens.

Any other name goes through upstream's bindings unchanged (e.g. ``terminus-2``).

Sandbox provider: ``HARBOR_ENV_TYPE`` (a Harbor ``EnvironmentType``, read by
upstream) or ``HARBOR_ENV_IMPORT_PATH`` (an environment class Harbor loads by
import path, for providers the pinned Harbor lacks, e.g.
``miles_ext.harbor.sail:SailEnvironment``). Upstream's ``_environment_config``
only knows the first, so this module replaces it on the loaded upstream
module; with no import path set it delegates unchanged.
Both mini-swe-agent harnesses use the same settings: mini.yaml templates, swebench.yaml shell env (in-sandbox cwd
``MSWEA_CWD``, else where Harbor starts the agent; the host harness keeps swebench.yaml's /testbed), ``HARBOR_AGENT_MAX_ITERATIONS``
steps, ``MSWEA_EXEC_TIMEOUT`` seconds per command, sampling params from the rollout.
Harbor installs the harness in each trial's sandbox; ``HARBOR_AGENT_SETUP_TIMEOUT_SEC``
(the recipe's ``agent.setup_timeout``) caps that install instead of Harbor's 360 s.
"""

import asyncio
import functools
import json
import logging
import os
import re
import threading
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from miles.rollout.agentic.session import openai_session_url

from miles_ext.harbor import context_exit, task_hash, trial_procs, upstream
from miles_ext.obs import sandbox_profile
from miles_ext.recipes import recipe as R

hf = upstream.harbor_agent_function()

logger = logging.getLogger(__name__)

SANDBOX_HARNESS, HOST_HARNESS, TERMINUS_SANDBOX_HARNESS = R.HARNESS_SANDBOX, R.HARNESS_HOST, R.HARNESS_TERMINUS_SANDBOX
# Harbor resolves agents it does not know by import path
_IMPORT_PATHS = {
    SANDBOX_HARNESS: "miles_ext.harbor.sandbox_agent:SandboxMiniSweAgent",
    HOST_HARNESS: "miles_ext.harbor.host_agent:HostMiniSweAgent",
    TERMINUS_SANDBOX_HARNESS: "miles_ext.harbor.terminus_sandbox:SandboxTerminus2",
}
COMPACT_ENV = "MILES_EXT_COMPACT_AT_TOKENS"


@functools.cache
def _swebench_environment() -> dict[str, Any]:
    """swebench.yaml's environment section; read once (every trial config uses it, on the rollout loop)."""
    from miles_ext.harbor.host_agent import load_builtin_config

    return load_builtin_config("benchmarks/swebench.yaml").get("environment", {})


@functools.cache
def _mini_swe_agent_version() -> str:
    return version("mini-swe-agent")


def _shell_config() -> dict[str, Any]:
    env_cfg = _swebench_environment()
    return {
        # "": mini-swe-agent runs commands where Harbor started it, the task's task.toml workdir
        # or else its image's WORKDIR; swebench.yaml's /testbed fits only SWE-bench images
        "cwd": os.getenv("MSWEA_CWD") or "",
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
        "version": _mini_swe_agent_version(),
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


def _terminus_sandbox_kwargs(session_url, api_key, sampling_params, model, max_seq_len) -> dict[str, Any]:
    """SandboxTerminus2's kwargs: upstream's terminus-2 kwargs (summarization and linear history from its
    HARBOR_TERMINUS_2_* env vars) with the model-call timeout the mini-swe-agent harnesses use."""
    terminus = hf.HARNESS_BINDINGS["terminus-2"].kwargs(
        session_url, api_key, _model_call_kwargs(sampling_params), model, max_seq_len
    )
    compact_at = os.getenv(COMPACT_ENV)
    return {"terminus": terminus, "compact_at_tokens": int(compact_at) if compact_at else None}


hf.HARNESS_BINDINGS[SANDBOX_HARNESS] = hf.HarnessBinding(kwargs=_sandbox_kwargs, env=hf._mini_swe_agent_env)
hf.HARNESS_BINDINGS[HOST_HARNESS] = hf.HarnessBinding(kwargs=_host_kwargs, env=lambda url, key: {})
hf.HARNESS_BINDINGS[TERMINUS_SANDBOX_HARNESS] = hf.HarnessBinding(
    kwargs=_terminus_sandbox_kwargs, env=hf.HARNESS_BINDINGS["terminus-2"].env
)
# harnesses that run inside the sandbox and reach the model through the session tunnel
_IN_SANDBOX = (SANDBOX_HARNESS, TERMINUS_SANDBOX_HARNESS)

_upstream_environment_config = hf._environment_config
# what upstream's _environment_config reads that an import-path provider does not honor here
_TYPE_ONLY_VARS = ("HARBOR_OVERRIDE_CPUS", "HARBOR_OVERRIDE_MEMORY_MB", "HARBOR_OVERRIDE_STORAGE_MB",
                   "HARBOR_CPU_ENFORCEMENT_POLICY", "HARBOR_MEMORY_ENFORCEMENT_POLICY")


def _environment_config():
    """Upstream's environment config, or one for ``HARBOR_ENV_IMPORT_PATH`` (an environment Harbor loads by path);
    with ``obs.sandbox_profile`` on, the same provider behind ext-040's timing wrapper."""
    config = _provider_config()
    return sandbox_profile.profiled_config(config) if sandbox_profile.enabled() else config


def _provider_config():
    import_path = os.getenv("HARBOR_ENV_IMPORT_PATH", "").strip()
    if not import_path:
        from miles_ext.harbor import modal_env

        config = _upstream_environment_config()
        if config.type is not None and config.type.value == modal_env.REPLACES_TYPE:  # ext-041
            return config.model_copy(update={"type": None, "import_path": modal_env.IMPORT_PATH})
        return config
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
        raise RuntimeError(f"{'/'.join(_IN_SANDBOX)} need the session tunnel (MILES_SESSION_PUBLIC_URL)")
    match = _SESSION_URL.match(session_url)
    if not match:
        raise RuntimeError(f"unexpected session URL {session_url!r}")
    port, path = match.groups()
    return f"{public.rstrip('/')}/{port}{path}"


def harness_for(metadata: dict[str, Any]) -> str:
    return os.getenv("MILES_EXT_HARNESS") or str(metadata.get("agent_name") or SANDBOX_HARNESS)


def build_trial_config(metadata: dict[str, Any], session_url: str, request_kwargs: dict[str, Any]):
    harness = harness_for(metadata)
    if harness in _IN_SANDBOX:
        session_url = public_session_url(session_url)
    config = hf.build_trial_config({**metadata, "agent_name": harness}, session_url, request_kwargs)
    if harness in _IN_SANDBOX:
        config.agent.extra_allowed_hosts = [*config.agent.extra_allowed_hosts, urlsplit(session_url).hostname]
    if harness == SANDBOX_HARNESS:
        # ContextExitAgent's window (context_exit.py)
        if max_seq_len := config.agent.kwargs.get("max_seq_len"):
            config.agent.env = {**config.agent.env, context_exit.ENV_LIMIT: str(max_seq_len)}
    if import_path := _IMPORT_PATHS.get(harness):
        config.agent.name = None
        config.agent.import_path = import_path
    if os.getenv(sandbox_profile.NO_VERIFIER_ENV) == "1":  # ext-040's benchmark times commands, not the task
        config.verifier.disable = True
    # upstream's agent function leaves Harbor's agent-install cap at 360 s x HARBOR_TIMEOUT_MULTIPLIER
    if setup_timeout := os.getenv("HARBOR_AGENT_SETUP_TIMEOUT_SEC"):
        config.agent.override_setup_timeout_sec = float(setup_timeout)
    return config


def _trial_modules() -> list[str]:
    """What a trial imports before its first sandbox call: the trial forkserver's preload (ext-042)."""
    modules = ["miles.rollout.agentic.agent_function", __name__, "harbor.trial.single_step", "harbor.trial.multi_step"]
    for path in (_IMPORT_PATHS.get(harness_for({})), os.getenv("HARBOR_ENV_IMPORT_PATH", "").strip()):
        if path:
            modules.append(path.partition(":")[0])
    return modules


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Run one Harbor trial for the sample in a process of its own; return its verdict as metadata."""
    if trial_procs.in_agent_process():
        return await run_trial(base_url, prompt, request_kwargs, metadata, **kwargs)
    from miles.dashboard.hooks import push_sandbox_reports

    return await trial_procs.call(
        run_trial,
        preload=_trial_modules(),
        on_child_reports=push_sandbox_reports,
        base_url=base_url,
        prompt=prompt,
        request_kwargs=request_kwargs,
        metadata=metadata,
        **kwargs,
    )


async def run_trial(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """``run``'s trial, in the calling process."""
    from harbor.trial.trial import Trial

    metadata = metadata or {}
    instance_id = metadata.get("instance_id")
    trial_timeout_s = hf._trial_timeout_s()

    # config errors fail every sample alike, so they raise instead of scoring 0
    hf._ensure_provider_key()
    config = build_trial_config(metadata, openai_session_url(base_url), request_kwargs or {})

    try:
        if config.task.path:  # off this loop, once per task per node (task_hash.py)
            await asyncio.to_thread(task_hash.warm, config.task.path)
        trial = await Trial.create(config)
        if sandbox_profile.enabled():
            sandbox_profile.note_session(trial.paths.trial_dir, base_url)
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
        if os.environ.get("MILES_EXT_CONTEXT_EXHAUSTED_SCORES_ZERO") == "1":
            out["reward"] = 0.0
    out["trial_dir"] = str(trial.paths.trial_dir)
    logger.info(f"Harbor trial {instance_id}: exit_status={out['exit_status']} reward={out['reward']}")
    return out


def prepare(args, data_source) -> None:
    """Rollout-setup hook (hook-005): start the trial forkserver and prewarm task hashes in a daemon thread."""
    dataset = getattr(data_source, "dataset", None)
    tasks_dir = os.getenv("HARBOR_TASKS_DIR")
    if dataset is None or not tasks_dir:
        return
    dirs = []
    for sample in dataset.samples:
        instance_id = str((sample.metadata or {}).get("instance_id", ""))
        try:
            dirs.append(hf._task_path(Path(tasks_dir), instance_id))
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"task_hash prewarm skips {instance_id!r}: {e}")
    modules = _trial_modules()

    def _run():
        try:
            trial_procs.start(modules)
        except Exception as e:
            logger.warning(f"prepare: starting the trial forkserver failed, the first trial starts it: {e!r}")
        task_hash.prewarm(dirs)

    threading.Thread(target=_run, name="harbor-prepare", daemon=True).start()
