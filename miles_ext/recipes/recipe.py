"""A run as data: a YAML recipe resolved against defaults and a model preset.

The recipe is the only input that changes between runs; the image (stack) and
this code stay fixed. ``miles_args`` turns a resolved recipe into the Miles argv
and ``harbor_env`` into the env vars the rollout workers read.

Paths in ``taskset`` may use environment variables (``$MILES_DATA``: where the
task sets live, outside this repo) and are otherwise relative to the recipe file. Container paths are fixed:
tasks at /inputs/tasks, the prompt file at /inputs/prompts.jsonl, weights at
/models/<repo name>, run outputs at /results/<run_id>.
"""

import copy
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any

import yaml

from miles_ext.recipes.presets import PRESETS

HARNESS_SANDBOX = "mini-swe-agent-sandbox"  # Harbor's mini-swe-agent inside the task sandbox
HARNESS_HOST = "mini-swe-agent-host"  # mini-swe-agent loop in the rollout worker, commands via exec
HARNESSES = (HARNESS_SANDBOX, HARNESS_HOST)
# Modal is a Harbor EnvironmentType; Sail is Harbor's provider vendored in
# miles_ext (the pinned Harbor predates it), loaded by import path
SANDBOX_PROVIDERS = ("modal", "sail")
SAIL_IMPORT_PATH = "miles_ext.harbor.sail:SailEnvironment"

GPUS_PER_NODE = 8
SESSION_PORT, SESSION_WORKERS = 30000, 32  # session servers listen on 30000..30031
PROXY_PORT = 29999  # session proxy, published by modal.forward for in-sandbox agents

MODELS = Path("/models")
RESULTS = Path("/results")
TASKS = Path("/inputs/tasks")
PROMPTS = Path("/inputs/prompts.jsonl")

DEFAULTS: dict[str, Any] = {
    "name": "",
    "model": {"preset": None},
    # cpu/memory are requests (Modal bills max(request, usage)); timeout caps a hung run
    "cluster": {"gpu": "H200", "nodes": 1, "train_gpus": 4, "cpu": 32, "memory_gib": 192, "timeout_hours": 6},
    "taskset": {"tasks_dir": None, "prompts": None, "harness": HARNESS_SANDBOX},
    "agent": {
        "max_steps": 50,
        "exec_timeout": 60,
        "timeout": 1800,
        "trial_timeout": 3600,
        "verifier_timeout": 900,
        "cwd": None,  # None: mini-swe-agent's swebench.yaml (/testbed)
    },
    "sandbox": {
        "provider": "modal",  # or "sail"
        # caps that hold even if the trainer dies: Modal terminates at either;
        # Sail terminates at lifetime_secs and may sleep (stop billing) after idle_secs
        "lifetime_secs": 7200,
        "idle_secs": 900,
        # Modal only: request Modal's minimum, keep the task's cpus/memory as hard
        # limits. Sail sizes are fixed shapes billed by observed usage, so the
        # task's resources only pick the size.
        "resource_policy": "limit",
        # sandbox app = app_prefix + run_id; callers that scope cleanup by name
        # (the platform's mdev-/mrl- prefixes) set their own prefix
        "app_prefix": "miles-fork-sbx-",
    },
    "algorithm": {
        "mode": "fully_async",  # or "sync" (colocated)
        "num_rollout": 20,
        "rollout_batch_size": 8,
        "n_samples": 8,
        "temperature": 1.0,
        "max_response_len": 4096,
        "max_seq_len": 32768,
        "lr": 1e-6,
        "weight_decay": 0.1,
        "adam_betas": [0.9, 0.98],
        "eps_clip": 0.2,
        "eps_clip_high": 0.28,
        "tis_clip": 2.0,
        "max_weight_staleness": 2,
        "sglang_mem_fraction": None,  # None: 0.85 fully async, 0.7 colocated
        "max_tokens_per_gpu": 32768,
    },
    "checkpoint": {"save_interval": 10},
    # W&B: one project per campaign, named by the recipe (fork/README.md, "W&B projects")
    "tracking": {"wandb_project": None, "wandb_entity": "mobiuslab", "tags": []},
    "extra_args": [],
}


def _merge(base: dict, override: dict, where: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if key not in base:
            raise KeyError(f"unknown recipe key {where}{key!r}")
        if isinstance(base[key], dict) and value is not None:
            out[key] = _merge(base[key], value, f"{where}{key}.")
        else:
            out[key] = value
    return out


def load(path: str | Path, overrides: dict | None = None) -> dict:
    """Read a recipe file, apply defaults and overrides, resolve local input paths."""
    path = Path(path).resolve()
    recipe = _merge(DEFAULTS, yaml.safe_load(path.read_text()) or {})
    recipe = _merge(recipe, overrides or {})
    for key in ("tasks_dir", "prompts"):
        if recipe["taskset"][key]:
            value = os.path.expandvars(recipe["taskset"][key])
            if "$" in value:
                raise ValueError(f"taskset.{key}: unset variable in {value!r} (export MILES_DATA)")
            recipe["taskset"][key] = str((path.parent / value).resolve())
    recipe["name"] = recipe["name"] or path.stem
    validate(recipe)
    return recipe


def validate(recipe: dict) -> None:
    preset = recipe["model"]["preset"]
    if preset not in PRESETS:
        raise ValueError(f"model.preset must be one of {sorted(PRESETS)}, got {preset!r}")
    if recipe["taskset"]["harness"] not in HARNESSES:
        raise ValueError(f"taskset.harness must be one of {HARNESSES}")
    sandbox = recipe["sandbox"]
    if sandbox["provider"] not in SANDBOX_PROVIDERS:
        raise ValueError(f"sandbox.provider must be one of {SANDBOX_PROVIDERS}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*-", sandbox["app_prefix"] or ""):
        raise ValueError(f"sandbox.app_prefix must be lowercase letters, digits and '-', ending in '-', "
                         f"got {sandbox['app_prefix']!r}")
    if sandbox["provider"] == "sail" and not 1 <= int(sandbox["idle_secs"]) <= 3600:
        raise ValueError("sandbox.idle_secs must be 1..3600 for sail (Sail's auto-sleep window)")
    if recipe["algorithm"]["mode"] not in ("fully_async", "sync"):
        raise ValueError("algorithm.mode must be fully_async or sync")
    cluster = recipe["cluster"]
    if recipe["algorithm"]["mode"] == "fully_async" and not 0 < cluster["train_gpus"] < cluster["nodes"] * GPUS_PER_NODE:
        raise ValueError("fully_async needs 0 < cluster.train_gpus < all GPUs (the rest serve rollouts)")
    if not recipe["taskset"]["tasks_dir"] or not recipe["taskset"]["prompts"]:
        raise ValueError("taskset.tasks_dir and taskset.prompts are required")
    project = recipe["tracking"]["wandb_project"]
    if not project or not re.fullmatch(r"miles-[a-z0-9][a-z0-9-]*", project):
        raise ValueError(f"tracking.wandb_project must name the campaign as miles-<campaign>, got {project!r}")
    if project in ARCHIVED_PROJECTS or project == SMOKE_PROJECT:
        raise ValueError(f"tracking.wandb_project {project!r} is reserved (archive or smoke runs)")


SMOKE_PROJECT = "miles-smoke"
SMOKE_PREFIXES = ("smoke-", "debug-")  # run ids that log to SMOKE_PROJECT instead of the recipe's project
ARCHIVED_PROJECTS = ("miles-swe-rl",)  # history from before this convention; no new runs


def wandb_project(recipe: dict, run_id: str) -> str:
    return SMOKE_PROJECT if run_id.startswith(SMOKE_PREFIXES) else recipe["tracking"]["wandb_project"]


def wandb_tags(recipe: dict) -> list[str]:
    """Tags every run carries; the launcher adds the fork release and commit."""
    return [f"recipe:{recipe['name']}", f"model:{recipe['model']['preset']}",
            f"sandbox:{recipe['sandbox']['provider']}", f"harness:{recipe['taskset']['harness']}",
            *recipe["tracking"]["tags"]]


SHAPE_KEYS = ("gpu", "nodes", "cpu", "memory_gib", "timeout_hours")


def shape(recipe: dict) -> dict[str, Any]:
    """The cluster fields Modal needs when the function is defined (decorator arguments)."""
    return {k: recipe["cluster"][k] for k in SHAPE_KEYS}


def needs_tunnel(recipe: dict) -> bool:
    return recipe["taskset"]["harness"] == HARNESS_SANDBOX


def model_dir(recipe: dict) -> Path:
    return MODELS / PRESETS[recipe["model"]["preset"]].dir_name


APP_NAME_MAX = 64  # Modal's limit on app names


def sandbox_app_name(recipe: dict, run_id: str) -> str:
    """Per-run app (Modal or Sail) for the task sandboxes, so cleanup touches only this run.

    Never truncated: a cut name could collide with another run's, and cleanup
    and reconcilers match on it."""
    name = recipe["sandbox"]["app_prefix"] + run_id
    if len(name) > APP_NAME_MAX:
        raise ValueError(f"sandbox app name {name!r} is longer than {APP_NAME_MAX}; shorten the run id or prefix")
    return name


def harbor_env(recipe: dict, run_id: str) -> dict[str, str]:
    """What every Harbor trial reads (header of upstream's harbor_agent_function)."""
    agent, sandbox = recipe["agent"], recipe["sandbox"]
    env = {
        **_sandbox_env(recipe, run_id),
        "HARBOR_TASKS_DIR": str(TASKS),
        "HARBOR_TRIALS_DIR": str(RESULTS / run_id / "trials"),
        "MILES_EXT_HARNESS": recipe["taskset"]["harness"],
        "AGENT_MODEL_NAME": "model",
        "AGENT_TIMEOUT": str(agent["timeout"]),
        "AGENT_TRIAL_TIMEOUT": str(agent["trial_timeout"]),
        "HARBOR_AGENT_MAX_ITERATIONS": str(agent["max_steps"]),
        "HARBOR_VERIFIER_TIMEOUT_SEC": str(agent["verifier_timeout"]),
        "HARBOR_TIMEOUT_MULTIPLIER": "1.0",
        "MSWEA_EXEC_TIMEOUT": str(agent["exec_timeout"]),
        "MSWEA_SILENT_STARTUP": "1",
        "MSWEA_COST_TRACKING": "ignore_errors",
    }
    if agent["cwd"]:
        env["MSWEA_CWD"] = agent["cwd"]
    return env


def _sandbox_env(recipe: dict, run_id: str) -> dict[str, str]:
    """How Harbor creates this run's sandboxes; each run gets its own app, so cleanup touches only it."""
    sandbox = recipe["sandbox"]
    if sandbox["provider"] == "sail":
        return {
            "HARBOR_ENV_IMPORT_PATH": SAIL_IMPORT_PATH,
            "HARBOR_ENV_KWARGS": json.dumps({
                "app_name": sandbox_app_name(recipe, run_id),
                "max_lifetime_secs": int(sandbox["lifetime_secs"]),
                "idle_secs": int(sandbox["idle_secs"]),
            }),
        }
    env = {
        "HARBOR_ENV_TYPE": "modal",
        # Harbor's Modal sandboxes otherwise live 24 h and outlive a killed trainer
        "HARBOR_ENV_KWARGS": json.dumps({
            "app_name": sandbox_app_name(recipe, run_id),
            "sandbox_timeout_secs": int(sandbox["lifetime_secs"]),
            "sandbox_idle_timeout_secs": int(sandbox["idle_secs"]),
        }),
        "MODAL_SANDBOX_V2": "1",
    }
    if policy := sandbox["resource_policy"]:
        env["HARBOR_CPU_ENFORCEMENT_POLICY"] = policy
        env["HARBOR_MEMORY_ENFORCEMENT_POLICY"] = policy
    return env


def miles_args(recipe: dict, run_id: str, resume: bool) -> list[str]:
    """The Miles argv (train.py / train_async.py) for a resolved recipe."""
    preset = PRESETS[recipe["model"]["preset"]]
    algo, cluster = recipe["algorithm"], recipe["cluster"]
    batch, n = int(algo["rollout_batch_size"]), int(algo["n_samples"])
    fully_async = algo["mode"] == "fully_async"
    total_gpus = int(cluster["nodes"]) * GPUS_PER_NODE
    ckpt_dir = RESULTS / run_id / "checkpoints"
    mem_fraction = algo["sglang_mem_fraction"] or (0.85 if fully_async else 0.7)
    beta1, beta2 = algo["adam_betas"]
    if fully_async:
        # Rollout engines keep generating while the trainer steps: one batch of
        # trajectories in flight, weights pushed after every step, in-flight
        # requests frozen in place across the push, groups more than
        # max_weight_staleness versions old dropped. TIS corrects the rest.
        placement = f"""
        --actor-num-nodes 1 --actor-num-gpus-per-node {cluster["train_gpus"]}
        --rollout-num-gpus {total_gpus - int(cluster["train_gpus"])} --num-gpus-per-node {GPUS_PER_NODE}
        --fully-async --update-weights-interval 1 --pause-generation-mode in_place
        --async-max-concurrent-samples {batch * n}
        --max-weight-staleness {algo["max_weight_staleness"]} --async-unused-samples-handler drop
        """
    else:
        placement = f"""
        --actor-num-nodes {cluster["nodes"]} --actor-num-gpus-per-node {GPUS_PER_NODE} --colocate
        """
    tracking = recipe["tracking"]
    args = shlex.split(
        f"""
        --train-backend {preset.backend}
        --hf-checkpoint {model_dir(recipe)}
        --save {ckpt_dir} --save-interval {recipe["checkpoint"]["save_interval"]}
        {f"--load {ckpt_dir}" if resume else ""}

        --prompt-data {PROMPTS} --input-key prompt --metadata-key metadata
        --rollout-shuffle --num-rollout {algo["num_rollout"]}
        --rollout-batch-size {batch} --n-samples-per-prompt {n} --global-batch-size {batch * n}
        --rollout-temperature {algo["temperature"]}
        --rollout-max-response-len {algo["max_response_len"]} --max-seq-len {algo["max_seq_len"]}

        --advantage-estimator grpo
        --use-tis --tis-clip {algo["tis_clip"]} --tis-clip-low 0.0
        --kl-coef 0 --entropy-coef 0
        --eps-clip {algo["eps_clip"]} --eps-clip-high {algo["eps_clip_high"]}
        --optimizer adam --lr {algo["lr"]} --lr-decay-style constant
        --weight-decay {algo["weight_decay"]} --adam-beta1 {beta1} --adam-beta2 {beta2}

        --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
        --custom-agent-function-path miles_ext.harbor.agent_function.run
        --custom-agent-function-mode inline
        --custom-rm-path miles_ext.harbor.reward.reward_func
        --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted
        --tito-model {preset.tito_model}
        --use-session-server --session-server-port {SESSION_PORT} --session-server-workers {SESSION_WORKERS}
        --pin-rollout-manager-to-head

        --rollout-num-gpus-per-engine 1
        --sglang-tool-call-parser {preset.tool_call_parser} --sglang-reasoning-parser {preset.reasoning_parser}
        --sglang-mem-fraction-static {mem_fraction}
        --sglang-decode-log-interval 1000
        --update-weight-buffer-size 536870912
        --gradient-checkpointing --attn-implementation flash_attention_2
        --use-dynamic-batch-size --max-tokens-per-gpu {algo["max_tokens_per_gpu"]}
        {placement}

        --use-wandb --wandb-project {wandb_project(recipe, run_id)} --wandb-group {run_id}
        {f"--wandb-team {tracking['wandb_entity']}" if tracking["wandb_entity"] else ""}
        """
    )
    if preset.chat_template_kwargs:
        args += ["--apply-chat-template-kwargs", json.dumps(preset.chat_template_kwargs)]
    args += ["--train-env-vars", json.dumps({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"})]
    return args + [str(a) for a in recipe["extra_args"]]


def entrypoint(recipe: dict) -> str:
    return "train_async.py" if recipe["algorithm"]["mode"] == "fully_async" else "train.py"
