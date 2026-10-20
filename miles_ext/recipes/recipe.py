"""A run as data: a YAML recipe resolved against defaults and a model preset.

The recipe is the only input that changes between runs; the image (stack) and
this code stay fixed. ``miles_args`` turns a resolved recipe into the Miles argv
and ``harbor_env`` into the env vars the rollout workers read.

Paths in ``taskset`` may use environment variables (``$MILES_DATA``: where the
task sets live, outside this repo) and are otherwise relative to the recipe file. Container paths are fixed:
tasks at /inputs/tasks, the prompt file at /inputs/prompts.jsonl, weights at
/models/<repo name>, run outputs at /results/<run_id>. ``taskset.tasks_volume``
(``<volume>:/<path>``) reads the tasks from a Modal volume instead of ``tasks_dir``,
mounted read-only at /datasets, so the launching machine holds only the prompt file.
"""

import copy
import json
import math
import os
import re
import shlex
from pathlib import Path
from typing import Any

import yaml

from miles.utils.external_utils.model_args_utils import load_model_args
from miles_ext.checkpoint.staged_save import STAGE_ENV
from miles_ext.recipes.presets import PRESETS

HARNESS_SANDBOX = "mini-swe-agent-sandbox"  # Harbor's mini-swe-agent inside the task sandbox
HARNESS_HOST = "mini-swe-agent-host"  # mini-swe-agent loop in the rollout worker, commands via exec
HARNESSES = (HARNESS_SANDBOX, HARNESS_HOST)
# Modal is a Harbor EnvironmentType; Sail is Harbor's provider vendored in
# miles_ext (the pinned Harbor predates it), loaded by import path
SANDBOX_PROVIDERS = ("modal", "sail")
BACKENDS = ("fsdp", "megatron")
SAIL_IMPORT_PATH = "miles_ext.harbor.sail:SailEnvironment"

GPUS_PER_NODE = 8
SESSION_PORT, SESSION_WORKERS = 30000, 32  # session servers listen on 30000..30031
PROXY_PORT = 29999  # session proxy, published by modal.forward for in-sandbox agents

MODELS = Path("/models")
RESULTS = Path("/results")
TASKS = Path("/inputs/tasks")
PROMPTS = Path("/inputs/prompts.jsonl")
DATASETS = Path("/datasets")  # taskset.tasks_volume's volume, read-only

DEFAULT_APP_PREFIX = "miles-mobius-"  # sandbox app = <prefix><run_id>
LEGACY_APP_PREFIXES = ("miles-fork-sbx-",)  # default before 2026-10-15; cleanup still looks there
# The Modal and Sail workspaces are shared with other teams (e.g. their mobius-<uuid>-N
# apps). Every sandbox app we create or clean up starts with one of these, and
# cleanup only ever addresses an exact app name, never a prefix or pattern.
OWN_APP_NAMESPACES = ("miles-", "rl-", "mdev-", "mstg-", "mrl-")  # rl-<env>-: the platform; m*-: its names before 2026-10-15
RUN_ID_RE = r"[A-Za-z0-9][A-Za-z0-9._-]*"

DEFAULTS: dict[str, Any] = {
    "name": "",
    # backend None: the preset's (fsdp); "megatron" needs the preset's megatron_model_type
    # revision: a commit sha of the preset's HF repo; set, the weights are read from the
    # shared cache key (miles_ext.platform.weights), unset from the preset's directory
    "model": {"preset": None, "backend": None, "revision": None},
    # cpu/memory are requests (Modal bills max(request, usage)); timeout caps a hung run
    # disk_gib: node-local disk per container (None: Modal's default quota, 512 GiB); needed when
    # optimizer.offload disk streams Adam's state to optimizer.disk_dir on that disk
    "cluster": {"gpu": "H200", "nodes": 1, "train_gpus": 4, "cpu": 32, "memory_gib": 192, "timeout_hours": 6,
                "disk_gib": None},
    # tasks_volume "<volume>:/<path>": the task dirs on a Modal volume, instead of tasks_dir
    "taskset": {"tasks_dir": None, "tasks_volume": None, "prompts": None, "harness": HARNESS_SANDBOX},
    "agent": {
        "max_steps": 50,
        "exec_timeout": 60,
        "timeout": 1800,
        "trial_timeout": 3600,
        "verifier_timeout": 900,
        # Harbor installs the harness in each trial's sandbox; its own cap (360 s) is too short
        # for a cold mini-swe-agent install on a busy provider
        "setup_timeout": 1800,
        "cwd": None,  # None: where Harbor starts the agent (task.toml workdir, else the image WORKDIR)
        # True: a trial that fills the context window scores 0 whatever its verifier says
        "context_exhausted_scores_zero": False,
        # "subproc": each trial in its own process, so one trial's blocking work cannot stall the others;
        # "inline": trials share the rollout event loop
        "function_mode": "subproc",
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
        "app_prefix": DEFAULT_APP_PREFIX,
        # multiplies each task's image build limit (Harbor's default 600 s); None: 1
        "build_timeout_multiplier": None,
        # set by a target with a short name id (rl-<env>-<id>-sbx); None: app_prefix + run_id
        "app_name": None,
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
        "tis_clip_low": 0.0,
        # MoE: replay the rollout's expert routing in training (R3)
        "routing_replay": False,
        # which finished groups train: "no_aborted", or "nonzero_std" (also drops groups whose
        # samples all scored the same; aborted groups are dropped before either filter)
        "group_filter": "no_aborted",
        "max_weight_staleness": 2,
        "sglang_mem_fraction": None,  # None: 0.85 fully async, 0.7 colocated
        # tokens packed per training micro-batch (a longer sample gets one of its own). FSDP holds
        # the micro-batch's full-vocabulary logits (~1 GiB per 1k tokens in fp32 at Qwen3.5's
        # 248k vocab): 32768 OOMed a 9B run on 4 H200s (smoke-r5-v1), so the default is 16384
        "max_tokens_per_gpu": 16384,
        "advantage_estimator": "grpo",
        # False: rewards are only centered on the group mean, not divided by its std (Dr. GRPO)
        "grpo_std_normalization": True,
        # True: the loss averages over all response tokens of the batch instead of per sample
        "per_token_loss": False,
        # "score_centering" (arXiv 2609.20807) replaces PPO clipping and --use-tis; score_centering_is weights it
        # ("tis" clips at tis_clip), top_logprobs is the sampler's candidates per generated token
        "loss": "policy_loss",
        "score_centering_is": "tis",
        "top_logprobs": 32,
        "kl_loss_coef": 0.0,  # > 0: KL loss against the base weights (FSDP then loads a reference copy)
        "entropy_coef": 0.0,
        "seed": None,  # None: Miles' default (1234)
    },
    # rollout engines: GPUs per engine (one SGLang server), the format they serve (a key of
    # the preset's precisions), context_length (None: the model's; set, a reply that would
    # overflow it is shortened), max_running_requests (None: SGLang's). expert_qat: the trainer
    # keeps its FP4 experts on the engine's grid (fp8_fp4 only, needs optimizer.offload disk)
    "rollout": {"gpus_per_engine": 1, "precision": None, "context_length": None, "max_running_requests": None,
                "expert_qat": False},
    # Megatron only. offload "disk": stream Adam's state to node-local disk between steps
    # (disk_dir on each trainer node; moment_dtype "bf16" halves the moments). grad_dtype: the
    # dtype gradients accumulate and reduce in; "bf16" halves the grad buffer (upstream's OOM lever)
    "optimizer": {"offload": None, "disk_dir": "/tmp/miles-optimizer", "moment_dtype": None, "grad_dtype": "fp32"},
    # Engines receive merged full weights; they need no native LoRA support.
    # target_modules uses Miles' selector syntax; None selects attention and MLP groups.
    "lora": {"rank": 0, "alpha": 32, "target_modules": None},
    # fully async rollout knobs (Miles' --async-* flags); None: Miles' default, or rollout_batch_size * n_samples
    # for max_concurrent_samples; retry_aborted_tries: fix-048, checkpoint_buffer: hook-008 (fork/patches/)
    "async": {"over_sampling_batch_size": None, "max_concurrent_samples": None, "unused_samples": "drop",
              "pause_generation": "in_place", "single_pass": False, "retry_aborted_tries": 1,
              "checkpoint_buffer": "log"},
    # keep: full saves kept while training (applied by the platform's retention);
    # keep_final_full: at the end keep the final full save (resume or continue
    # training) or only its HF weights; save_hf: Megatron writes HF weights at each save
    # save_optimizer False: saves hold weights only and a resume starts Adam's moments afresh
    # save_rollout_data: each batch the trainer receives goes to <run>/rollout_data/<rollout_id>.pt,
    # so a trainer failure replays on trainer nodes alone (Miles --load-debug-rollout-data)
    # stage_dir (Megatron only): each node saves to this node-local dir and uploads to the results
    # volume in the background (miles_ext.checkpoint.staged_save); None: saves go to the volume directly
    "checkpoint": {"save_interval": 10, "keep": 1, "keep_final_full": True, "save_hf": False,
                   "save_optimizer": True, "save_rollout_data": False, "stage_dir": None},
    # Megatron only: tensor / pipeline / context parallel sizes; data parallel is
    # what is left of the trainer GPUs (cluster.train_gpus, or all GPUs when colocated).
    # ep: expert parallel size for MoE models (None: Miles' default)
    "parallel": {"tp": 1, "pp": 1, "cp": 1, "ep": None},
    # W&B: one project per campaign, named by the recipe (fork/README.md, "W&B projects")
    # wandb_name: the W&B display name (the platform's "<short id> · <run name>"); None: the run id
    "tracking": {"wandb_project": None, "wandb_entity": "mobiuslab", "wandb_name": None, "tags": []},
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


def from_dict(spec: dict, overrides: dict | None = None) -> dict:
    """A recipe from a dict (the platform's spec), with defaults and overrides; inputs stay where they are."""
    return _merge(_merge(DEFAULTS, spec or {}), overrides or {})


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


GPU_TYPES = ("H100", "H200", "B200", "B300")
# Miles' choices at our pin, minus what this stack cannot run: "ppo" needs a critic the recipe
# does not build; pause "abort" and --partial-rollout are rejected by Miles in fully async mode.
# Miles requires --normalize-advantages with the reinforce_plus_plus estimators; miles_args adds it
ADVANTAGE_ESTIMATORS = ("grpo", "gspo", "reinforce_plus_plus", "reinforce_plus_plus_baseline")
UNUSED_SAMPLES = ("drop", "retry")
CHECKPOINT_BUFFERS = ("log", "prompts", "groups")
GROUP_LOG_BUFFER = "miles_ext.rollout.group_log.GroupLogDataBuffer"
PAUSE_GENERATION = ("in_place", "retract")
GROUP_FILTERS = {
    "no_aborted": "miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted",
    "nonzero_std": "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std",
}
# optimizer.grad_dtype -> the Megatron flag that sets the dtype gradients accumulate and reduce in
_GRAD_DTYPE_FLAGS = {"fp32": "--accumulate-allreduce-grads-in-fp32", "bf16": "--grad-reduce-in-bf16"}
FP8_FP4 = "fp8_fp4"  # the released DeepSeek V4.1 format: FP8 dense, MXFP4 experts (ext-007)

# path -> rule; every other leaf takes its type from DEFAULTS. Keys: type ("int", "number", "str",
# "bool", "list"), nullable, min, max, enum.
FIELD_RULES: dict[str, dict] = {
    "cluster.gpu": {"type": "str", "enum": GPU_TYPES},
    "cluster.nodes": {"type": "int", "min": 1, "max": 16},
    "cluster.train_gpus": {"type": "int", "min": 1},
    "cluster.cpu": {"type": "number", "min": 1},
    "cluster.memory_gib": {"type": "int", "min": 1},
    "cluster.timeout_hours": {"type": "number", "min": 0.1, "max": 24},
    "cluster.disk_gib": {"type": "int", "nullable": True, "min": 1, "max": 3072},  # Modal's ephemeral_disk cap
    "model.preset": {"type": "str", "enum": tuple(sorted(PRESETS))},
    "model.backend": {"type": "str", "nullable": True, "enum": BACKENDS},
    "model.revision": {"type": "str", "nullable": True},
    "taskset.tasks_dir": {"type": "str", "nullable": True},
    "taskset.tasks_volume": {"type": "str", "nullable": True},
    "taskset.prompts": {"type": "str", "nullable": True},
    "taskset.harness": {"type": "str", "enum": HARNESSES},
    "agent.max_steps": {"type": "int", "min": 1},
    "agent.exec_timeout": {"type": "int", "min": 1},
    "agent.timeout": {"type": "int", "min": 1},
    "agent.trial_timeout": {"type": "int", "min": 1},
    "agent.verifier_timeout": {"type": "int", "min": 1},
    "agent.setup_timeout": {"type": "int", "min": 1},
    "agent.cwd": {"type": "str", "nullable": True},
    "agent.function_mode": {"type": "str", "enum": ("subproc", "inline")},
    "sandbox.provider": {"type": "str", "enum": SANDBOX_PROVIDERS},
    "sandbox.lifetime_secs": {"type": "int", "min": 60},
    "sandbox.idle_secs": {"type": "int", "min": 1},
    "sandbox.resource_policy": {"type": "str", "enum": ("limit", "request")},
    "sandbox.build_timeout_multiplier": {"type": "number", "nullable": True, "min": 1},
    "algorithm.mode": {"type": "str", "enum": ("fully_async", "sync")},
    "algorithm.num_rollout": {"type": "int", "min": 1},
    "algorithm.rollout_batch_size": {"type": "int", "min": 1},
    "algorithm.n_samples": {"type": "int", "min": 1},
    "algorithm.temperature": {"type": "number", "min": 0},
    "algorithm.max_response_len": {"type": "int", "min": 1},
    "algorithm.max_seq_len": {"type": "int", "min": 1},
    "algorithm.lr": {"type": "number", "min": 0, "exclusive_min": True},
    "algorithm.weight_decay": {"type": "number", "min": 0},
    "algorithm.eps_clip": {"type": "number", "min": 0},
    "algorithm.eps_clip_high": {"type": "number", "min": 0},
    "algorithm.tis_clip": {"type": "number", "min": 0},
    "algorithm.tis_clip_low": {"type": "number", "min": 0},
    "algorithm.group_filter": {"type": "str", "enum": tuple(GROUP_FILTERS)},
    "algorithm.max_weight_staleness": {"type": "int", "min": 0},
    "algorithm.sglang_mem_fraction": {"type": "number", "nullable": True, "min": 0.05, "max": 0.95},
    "algorithm.max_tokens_per_gpu": {"type": "int", "min": 1},
    "algorithm.advantage_estimator": {"type": "str", "enum": ADVANTAGE_ESTIMATORS},
    "algorithm.loss": {"type": "str", "enum": ("policy_loss", "score_centering")},
    "algorithm.score_centering_is": {"type": "str", "enum": ("none", "tis", "mis")},
    "algorithm.top_logprobs": {"type": "int", "min": 1},
    "algorithm.kl_loss_coef": {"type": "number", "min": 0},
    "algorithm.entropy_coef": {"type": "number", "min": 0},
    "algorithm.seed": {"type": "int", "nullable": True},
    "rollout.gpus_per_engine": {"type": "int", "min": 1, "max": 8},
    "rollout.precision": {"type": "str", "nullable": True},
    "rollout.context_length": {"type": "int", "nullable": True, "min": 1},
    "rollout.max_running_requests": {"type": "int", "nullable": True, "min": 1},
    "optimizer.offload": {"type": "str", "nullable": True, "enum": ("disk",)},
    "optimizer.disk_dir": {"type": "str"},
    "optimizer.moment_dtype": {"type": "str", "nullable": True, "enum": ("bf16",)},
    "optimizer.grad_dtype": {"type": "str", "enum": tuple(_GRAD_DTYPE_FLAGS)},
    "lora.rank": {"type": "int", "min": 0},
    "lora.alpha": {"type": "int", "min": 1},
    "lora.target_modules": {"type": "str", "nullable": True},
    "async.over_sampling_batch_size": {"type": "int", "nullable": True, "min": 1},
    "async.max_concurrent_samples": {"type": "int", "nullable": True, "min": 1},
    "async.unused_samples": {"type": "str", "enum": UNUSED_SAMPLES},
    "async.retry_aborted_tries": {"type": "int", "min": 0},
    "async.checkpoint_buffer": {"type": "str", "enum": CHECKPOINT_BUFFERS},
    "async.pause_generation": {"type": "str", "enum": PAUSE_GENERATION},
    "checkpoint.save_interval": {"type": "int", "min": 1},
    "checkpoint.keep": {"type": "int", "min": 1},
    "checkpoint.stage_dir": {"type": "str", "nullable": True},
    "parallel.tp": {"type": "int", "min": 1},
    "parallel.pp": {"type": "int", "min": 1},
    "parallel.cp": {"type": "int", "min": 1},
    "parallel.ep": {"type": "int", "nullable": True, "min": 1},
    "tracking.wandb_project": {"type": "str", "nullable": True},
    "tracking.wandb_entity": {"type": "str", "nullable": True},
    "tracking.wandb_name": {"type": "str", "nullable": True},
}
_TYPES = {"int": (int,), "number": (int, float), "str": (str,), "bool": (bool,), "list": (list,)}


def _leaves(d: dict, prefix: str = ""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _leaves(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def field_rule(path: str, default) -> dict | None:
    if path in FIELD_RULES:
        return FIELD_RULES[path]
    for name, types in (("bool", (bool,)), ("int", (int,)), ("number", (float,)), ("str", (str,)), ("list", (list,))):
        if isinstance(default, types):
            return {"type": name}
    return None  # None default with no rule: any value


def field_errors(recipe: dict) -> list[dict]:
    """Type, range and choice errors per field, as ``{path, message}`` (before any check that
    computes with the values, so a string where a number belongs is reported, not raised)."""
    errors = []
    for path, default in _leaves(DEFAULTS):
        rule = field_rule(path, default)
        if rule is None:
            continue
        value = recipe
        for key in path.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        if value is None:
            if not rule.get("nullable") and default is not None:
                errors.append({"path": path, "message": f"{path} is required"})
            continue
        if isinstance(value, bool) and rule["type"] != "bool" or not isinstance(value, _TYPES[rule["type"]]):
            errors.append({"path": path, "message": f"{path} must be {rule['type']}, got {value!r}"})
            continue
        if "enum" in rule and value not in rule["enum"]:
            errors.append({"path": path, "message": f"{path} must be one of {list(rule['enum'])}, got {value!r}"})
        if "min" in rule and (value < rule["min"] or rule.get("exclusive_min") and value == rule["min"]):
            op = ">" if rule.get("exclusive_min") else ">="
            errors.append({"path": path, "message": f"{path} must be {op} {rule['min']}, got {value!r}"})
        if "max" in rule and value > rule["max"]:
            errors.append({"path": path, "message": f"{path} must be <= {rule['max']}, got {value!r}"})
    return errors


def validate(recipe: dict, local_inputs: bool = True) -> None:
    """``local_inputs=False``: the taskset is on the results volume (platform runs), not local paths.
    Raises ValueError with the first problem (``field_errors`` lists every field's)."""
    if errors := field_errors(recipe):
        raise ValueError(errors[0]["message"])
    sandbox = recipe["sandbox"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*-", sandbox["app_prefix"] or ""):
        raise ValueError(f"sandbox.app_prefix must be lowercase letters, digits and '-', ending in '-', "
                         f"got {sandbox['app_prefix']!r}")
    if not sandbox["app_prefix"].startswith(OWN_APP_NAMESPACES):
        raise ValueError(f"sandbox.app_prefix must start with one of {OWN_APP_NAMESPACES} (shared workspace), "
                         f"got {sandbox['app_prefix']!r}")
    if sandbox.get("app_name") is not None:
        check_own_app_name(sandbox["app_name"])
        if len(sandbox["app_name"]) > APP_NAME_MAX:
            raise ValueError(f"sandbox.app_name {sandbox['app_name']!r} is longer than {APP_NAME_MAX}")
    if sandbox["provider"] == "sail" and not 1 <= int(sandbox["idle_secs"]) <= 3600:
        raise ValueError("sandbox.idle_secs must be 1..3600 for sail (Sail's auto-sleep window)")
    cluster = recipe["cluster"]
    if recipe["algorithm"]["mode"] == "fully_async" and not 0 < cluster["train_gpus"] < cluster["nodes"] * GPUS_PER_NODE:
        raise ValueError("fully_async needs 0 < cluster.train_gpus < all GPUs (the rest serve rollouts)")
    algo, aio = recipe["algorithm"], recipe["async"]
    if aio["max_concurrent_samples"] is not None and aio["max_concurrent_samples"] < algo["n_samples"]:
        raise ValueError("async.max_concurrent_samples must be >= algorithm.n_samples (Miles asserts it)")
    if aio["over_sampling_batch_size"] is not None and aio["over_sampling_batch_size"] < algo["rollout_batch_size"]:
        raise ValueError("async.over_sampling_batch_size must be >= algorithm.rollout_batch_size (Miles asserts it)")
    if aio["single_pass"] and algo["mode"] != "fully_async":
        raise ValueError("async.single_pass needs algorithm.mode fully_async (Miles asserts it)")
    _validate_trainer(recipe)
    _validate_rollout(recipe)
    taskset = recipe["taskset"]
    if taskset["tasks_dir"] and taskset["tasks_volume"]:
        raise ValueError("taskset: set tasks_dir or tasks_volume, not both")
    tasks_volume(recipe)  # its format
    if local_inputs and (not (taskset["tasks_dir"] or taskset["tasks_volume"]) or not taskset["prompts"]):
        raise ValueError("taskset.prompts and one of taskset.tasks_dir, taskset.tasks_volume are required")
    ckpt = recipe["checkpoint"]
    if ckpt["save_hf"] and backend(recipe) != "megatron":
        raise ValueError("checkpoint.save_hf is Megatron-only; FSDP runs get HF weights from finalize_hf")
    if ckpt["stage_dir"] and backend(recipe) != "megatron":
        raise ValueError("checkpoint.stage_dir is Megatron-only")
    revision = recipe["model"]["revision"]
    if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
        raise ValueError(f"model.revision must be a full commit sha, got {revision!r}")
    project = recipe["tracking"]["wandb_project"]
    if not project or not re.fullmatch(r"miles-[a-z0-9][a-z0-9-]*", project):
        raise ValueError(f"tracking.wandb_project must name the campaign as miles-<campaign>, got {project!r}")
    if project in ARCHIVED_PROJECTS or project == SMOKE_PROJECT:
        raise ValueError(f"tracking.wandb_project {project!r} is reserved (archive or smoke runs)")


def rollout_precision(recipe: dict) -> str:
    return recipe["rollout"]["precision"] or next(iter(PRESETS[recipe["model"]["preset"]].precisions))


def _validate_rollout(recipe: dict) -> None:
    preset, roll, cluster = PRESETS[recipe["model"]["preset"]], recipe["rollout"], recipe["cluster"]
    precision, gpus = rollout_precision(recipe), int(roll["gpus_per_engine"])
    if precision not in preset.precisions:
        raise ValueError(
            f"rollout.precision must be one of {list(preset.precisions)} for this model, got {precision!r}")
    if GPUS_PER_NODE % gpus:
        raise ValueError(
            f"rollout.gpus_per_engine {gpus} must divide {GPUS_PER_NODE} so an engine stays inside a node")
    rollout_gpus = cluster["nodes"] * GPUS_PER_NODE - cluster["train_gpus"]
    if recipe["algorithm"]["mode"] == "fully_async" and rollout_gpus % gpus:
        raise ValueError(f"the rollout GPUs (all GPUs minus cluster.train_gpus) must split into engines of {gpus}")
    if precision == FP8_FP4 and (backend(recipe) != "megatron" or not preset.train_dir_name):
        raise ValueError(f"rollout.precision {FP8_FP4} needs the Megatron trainer and a BF16 train copy in the preset")
    if precision == FP8_FP4 and recipe["checkpoint"]["save_hf"]:
        # Megatron's HF export writes under --hf-checkpoint's config, which here is the quantized one
        raise ValueError(f"checkpoint.save_hf is not supported with rollout.precision {FP8_FP4}")
    if roll["expert_qat"]:
        if precision != FP8_FP4 or recipe["optimizer"]["offload"] != "disk":
            raise ValueError(f"rollout.expert_qat needs rollout.precision {FP8_FP4} and optimizer.offload disk "
                             "(Miles projects the experts in the disk-streamed Adam)")
        par = megatron_parallel(recipe)
        if par["ep"] != par["tp"] * par["cp"] * par["dp"]:
            raise ValueError("rollout.expert_qat needs expert data parallel size 1: parallel.ep = tp * cp * dp")


def backend(recipe: dict) -> str:
    return recipe["model"]["backend"] or PRESETS[recipe["model"]["preset"]].backend


def trainer_gpus(recipe: dict) -> int:
    cluster = recipe["cluster"]
    if recipe["algorithm"]["mode"] == "fully_async":
        return int(cluster["train_gpus"])
    return int(cluster["nodes"]) * GPUS_PER_NODE


def trainer_layout(recipe: dict) -> tuple[int, int]:
    """(--actor-num-nodes, --actor-num-gpus-per-node). Miles hands the trainer the
    first GPUs in (node IP, GPU id) order, so a trainer of more than 8 GPUs spans
    nodes: 12 GPUs on two nodes are 8 on the first and 4 on the second, declared
    as 2 x 6 (Miles only uses the product to size the placement group)."""
    gpus = trainer_gpus(recipe)
    nodes = math.ceil(gpus / GPUS_PER_NODE)
    return nodes, gpus // nodes


def megatron_parallel(recipe: dict) -> dict[str, int]:
    """tp, pp, cp and ep (None: Miles' default of 1) from the recipe, and dp: the trainer GPUs they leave."""
    par = {k: int(v or 1) for k, v in recipe["parallel"].items()}
    return {**par, "dp": trainer_gpus(recipe) // (par["tp"] * par["pp"] * par["cp"])}


def _model_dims(model_type: str) -> dict[str, int]:
    tokens = shlex.split(load_model_args(model_type))
    flags = ("--num-layers", "--num-attention-heads", "--num-query-groups")
    return {f: int(tokens[tokens.index(f) + 1]) for f in flags if f in tokens}


def _validate_trainer(recipe: dict) -> None:
    """Layout errors a Megatron run would otherwise hit minutes into a cluster start."""
    name = backend(recipe)
    if name not in BACKENDS:
        raise ValueError(f"model.backend must be one of {BACKENDS}, got {name!r}")
    nodes, per_node = trainer_layout(recipe)
    if nodes * per_node != trainer_gpus(recipe):
        raise ValueError(f"{trainer_gpus(recipe)} trainer GPUs do not split evenly over {nodes} nodes")
    if name != "megatron":
        if recipe["parallel"] != DEFAULTS["parallel"]:
            raise ValueError("parallel.* applies to model.backend: megatron only")
        if recipe["optimizer"] != DEFAULTS["optimizer"] or not recipe["checkpoint"]["save_optimizer"]:
            raise ValueError("optimizer.* and checkpoint.save_optimizer apply to model.backend: megatron only")
        if recipe["lora"]["rank"]:
            raise ValueError("lora.* applies to model.backend: megatron only")
        return
    if recipe["optimizer"]["offload"] == "disk" and not recipe["cluster"]["disk_gib"]:
        # Modal's default per-container quota (512 GiB) is far below a large model's Adam state
        raise ValueError("optimizer.offload disk needs cluster.disk_gib (node-local disk for Adam's state)")
    if recipe["lora"]["rank"]:
        _validate_lora(recipe)
    model_type = PRESETS[recipe["model"]["preset"]].megatron_model_type
    if not model_type:
        raise ValueError(f"preset {recipe['model']['preset']!r} has no megatron_model_type")
    par, dims = megatron_parallel(recipe), _model_dims(model_type)
    tp, pp, cp, ep, dp = par["tp"], par["pp"], par["cp"], par["ep"], par["dp"]
    samples = int(recipe["algorithm"]["rollout_batch_size"]) * int(recipe["algorithm"]["n_samples"])
    checks = [
        (dp >= 1 and tp * pp * cp * dp == trainer_gpus(recipe),
         f"tp*pp*cp = {tp * pp * cp} must divide the {trainer_gpus(recipe)} trainer GPUs"),
        (GPUS_PER_NODE % tp == 0, f"tp {tp} must divide {GPUS_PER_NODE} so TP groups stay inside a node"),
        # Miles splits each step's samples into equal parts per DP rank
        (samples % dp == 0, f"rollout_batch_size * n_samples = {samples} must divide by dp {dp}"),
        (dims.get("--num-attention-heads", tp) % tp == 0, f"tp {tp} must divide the attention heads"),
        (dims.get("--num-query-groups", tp) % tp == 0, f"tp {tp} must divide the KV groups"),
        (dims.get("--num-layers", pp) % pp == 0, f"pp {pp} must divide the {dims.get('--num-layers')} layers"),
        # Megatron builds expert groups inside one pipeline stage
        ((tp * cp * dp) % ep == 0, f"ep {ep} must divide the {tp * cp * dp} GPUs of a pipeline stage"),
    ]
    for ok, message in checks:
        if not ok:
            raise ValueError(f"parallel: {message}")


def _validate_lora(recipe: dict) -> None:
    if recipe["algorithm"]["mode"] != "fully_async":
        raise ValueError("lora needs algorithm.mode fully_async: the merged sync reads the live trainer weights")
    if recipe["optimizer"]["offload"]:
        raise ValueError(
            "lora trains only the adapters; drop optimizer.offload (Miles refuses the disk Adam with LoRA)"
        )
    if recipe["rollout"]["expert_qat"]:
        raise ValueError("lora leaves the routed experts frozen; drop rollout.expert_qat")


def _lora_args(recipe: dict) -> str:
    lora = recipe["lora"]
    if not lora["rank"]:
        return ""
    args = f" --lora-rank {lora['rank']} --lora-alpha {lora['alpha']} --lora-merged-sync"
    if lora["target_modules"]:
        args += f" --target-modules {lora['target_modules']}"
    if spec := PRESETS[recipe["model"]["preset"]].lora_spec_path:
        args += f" --custom-lora-spec-path {spec}"
    return args


SMOKE_PROJECT = "miles-smoke"
SMOKE_PREFIXES = ("smoke-", "debug-")  # run ids that log to SMOKE_PROJECT instead of the recipe's project
ARCHIVED_PROJECTS = ("miles-swe-rl",)  # history from before this convention; no new runs


def wandb_project(recipe: dict, run_id: str) -> str:
    return SMOKE_PROJECT if run_id.startswith(SMOKE_PREFIXES) else recipe["tracking"]["wandb_project"]


def wandb_tags(recipe: dict) -> list[str]:
    """Tags every run carries; the launcher adds ``miles-mobius:``, ``upstream:`` and ``commit:``."""
    return [f"recipe:{recipe['name']}", f"model:{recipe['model']['preset']}",
            f"sandbox:{recipe['sandbox']['provider']}", f"harness:{recipe['taskset']['harness']}",
            *recipe["tracking"]["tags"]]


SHAPE_KEYS = ("gpu", "nodes", "cpu", "memory_gib", "timeout_hours", "disk_gib")


def shape(recipe: dict) -> dict[str, Any]:
    """The cluster fields Modal needs when the function is defined (decorator arguments)."""
    return {k: recipe["cluster"][k] for k in SHAPE_KEYS}


def needs_tunnel(recipe: dict) -> bool:
    return recipe["taskset"]["harness"] == HARNESS_SANDBOX


def model_dir(recipe: dict) -> Path:
    preset = PRESETS[recipe["model"]["preset"]]
    if revision := recipe["model"].get("revision"):
        return MODELS / f"{preset.hf_repo.replace('/', '--')}/{revision}"  # miles_ext.platform.weights key
    return MODELS / preset.dir_name


APP_NAME_MAX = 64  # Modal's limit on app names


def sandbox_app_name(recipe: dict, run_id: str) -> str:
    """Per-run app (Modal or Sail) for the task sandboxes, so cleanup touches only this run.

    Never truncated: a cut name could collide with another run's, and cleanup
    and reconcilers match on it."""
    if not re.fullmatch(RUN_ID_RE, run_id):
        raise ValueError(f"run id {run_id!r}: letters, digits, '.', '_', '-', starting with a letter or digit")
    name = recipe["sandbox"].get("app_name") or recipe["sandbox"]["app_prefix"] + run_id
    if len(name) > APP_NAME_MAX:
        raise ValueError(f"sandbox app name {name!r} is longer than {APP_NAME_MAX}; shorten the run id or prefix")
    return check_own_app_name(name)


def check_own_app_name(name: str) -> str:
    """Raise unless ``name`` is one exact app name in our namespaces: cleanup terminates
    everything in the app it names, on workspaces shared with other teams."""
    prefix = next((p for p in OWN_APP_NAMESPACES if name.startswith(p)), None)
    if prefix is None or not re.fullmatch(RUN_ID_RE, name[len(prefix):]):
        raise ValueError(f"{name!r} is not one of our sandbox app names "
                         f"(<{'|'.join(OWN_APP_NAMESPACES)}>..., letters, digits, '.', '_', '-')")
    return name


def sandbox_app_names_for_cleanup(recipe: dict, run_id: str) -> list[str]:
    """The run's sandbox app plus the names earlier fork versions gave the same run id,
    so a run started before a rename and resumed after it still leaves nothing behind."""
    names = [sandbox_app_name(recipe, run_id)]
    if recipe["sandbox"]["app_prefix"] == DEFAULT_APP_PREFIX and not recipe["sandbox"].get("app_name"):
        names += [check_own_app_name(p + run_id) for p in LEGACY_APP_PREFIXES if len(p + run_id) <= APP_NAME_MAX]
    return names


def tasks_volume(recipe: dict) -> tuple[str, str] | None:
    """``taskset.tasks_volume`` as (volume name, path in it), or None when the tasks come from tasks_dir."""
    spec = recipe["taskset"].get("tasks_volume")
    if not spec:
        return None
    m = re.fullmatch(r"([A-Za-z0-9._-]+):(/[A-Za-z0-9._/-]*)", spec)
    if not m or ".." in m[2].split("/"):
        raise ValueError(f"taskset.tasks_volume must be <volume>:/<path>, got {spec!r}")
    if not m[1].startswith(OWN_APP_NAMESPACES):
        raise ValueError(f"taskset.tasks_volume: volume {m[1]!r} must start with one of {OWN_APP_NAMESPACES}")
    return m[1], m[2]


def tasks_path(recipe: dict) -> Path:
    """Where the task dirs are in the container."""
    volume = tasks_volume(recipe)
    return DATASETS / volume[1].lstrip("/") if volume else TASKS


def harbor_env(recipe: dict, run_id: str) -> dict[str, str]:
    """What every Harbor trial reads (header of upstream's harbor_agent_function)."""
    agent = recipe["agent"]
    env = {
        **_sandbox_env(recipe, run_id),
        "HARBOR_TASKS_DIR": str(tasks_path(recipe)),
        "HARBOR_TRIALS_DIR": str(RESULTS / run_id / "trials"),
        "MILES_EXT_HARNESS": recipe["taskset"]["harness"],
        "AGENT_MODEL_NAME": "model",
        "AGENT_TIMEOUT": str(agent["timeout"]),
        "AGENT_TRIAL_TIMEOUT": str(agent["trial_timeout"]),
        "HARBOR_AGENT_MAX_ITERATIONS": str(agent["max_steps"]),
        "HARBOR_VERIFIER_TIMEOUT_SEC": str(agent["verifier_timeout"]),
        "HARBOR_AGENT_SETUP_TIMEOUT_SEC": str(agent["setup_timeout"]),
        "HARBOR_TIMEOUT_MULTIPLIER": "1.0",
        "MSWEA_EXEC_TIMEOUT": str(agent["exec_timeout"]),
        "MSWEA_SILENT_STARTUP": "1",
        "MSWEA_COST_TRACKING": "ignore_errors",
    }
    if multiplier := recipe["sandbox"]["build_timeout_multiplier"]:
        env["HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER"] = str(multiplier)
    if agent["cwd"]:
        env["MSWEA_CWD"] = agent["cwd"]
    if agent["context_exhausted_scores_zero"]:
        env["MILES_EXT_CONTEXT_EXHAUSTED_SCORES_ZERO"] = "1"
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
    actor_nodes, actor_gpus_per_node = trainer_layout(recipe)
    beta1, beta2 = algo["adam_betas"]
    aio = recipe["async"]
    kl_loss = float(algo["kl_loss_coef"]) > 0
    if fully_async:
        # Rollout engines keep generating while the trainer steps: one batch of
        # trajectories in flight, weights pushed after every step, in-flight
        # requests frozen in place across the push, groups more than
        # max_weight_staleness versions old dropped. TIS corrects the rest.
        # Miles defaults in_place to a radix cache namespaced per rollout call,
        # which needs --use-session-server v2; on v1 the argv passes
        # --no-namespaced-radix-cache and keeps one shared cache, as before
        # upstream af4413b74.
        placement = f"""
        --actor-num-nodes {actor_nodes} --actor-num-gpus-per-node {actor_gpus_per_node}
        --rollout-num-gpus {total_gpus - int(cluster["train_gpus"])} --num-gpus-per-node {GPUS_PER_NODE}
        --fully-async --update-weights-interval 1 --pause-generation-mode {aio["pause_generation"]}
        --async-max-concurrent-samples {aio["max_concurrent_samples"] or batch * n}
        --max-weight-staleness {algo["max_weight_staleness"]} --async-unused-samples-handler {aio["unused_samples"]}
        --async-checkpoint-buffer {aio["checkpoint_buffer"]}
        {f"--custom-async-data-buffer-path {GROUP_LOG_BUFFER}" if aio["checkpoint_buffer"] == "log" else ""}
        --async-retry-aborted-tries {aio["retry_aborted_tries"]}
        {f"--over-sampling-batch-size {aio['over_sampling_batch_size']}" if aio["over_sampling_batch_size"] else ""}
        {"--async-single-pass" if aio["single_pass"] else ""}
        """
    else:
        placement = f"""
        --actor-num-nodes {cluster["nodes"]} --actor-num-gpus-per-node {GPUS_PER_NODE} --colocate
        """
    tracking = recipe["tracking"]
    # Miles turns the post-sync engine weight checksum audit on whenever an
    # event dir is set (--save gives one). It crashed the DSV4.1 FP8+FP4
    # engines (CUDA illegal address) in dsv41-tb4-overfit16-1, so it stays off.
    args = shlex.split(
        f"""
        --train-backend {backend(recipe)}
        --hf-checkpoint {model_dir(recipe)}
        --save {ckpt_dir} --save-interval {recipe["checkpoint"]["save_interval"]}
        --no-log-inference-engine-weight-checksums
        {f"--save-hf {RESULTS / run_id / 'hf' / 'iter_{rollout_id:07d}'}" if recipe["checkpoint"]["save_hf"] else ""}
        {f"--load {ckpt_dir}" if resume else ""}

        --prompt-data {PROMPTS} --input-key prompt --metadata-key metadata
        --rollout-shuffle --num-rollout {algo["num_rollout"]}
        --rollout-batch-size {batch} --n-samples-per-prompt {n} --global-batch-size {batch * n}
        --rollout-temperature {algo["temperature"]}
        --rollout-max-response-len {algo["max_response_len"]} --max-seq-len {algo["max_seq_len"]}

        --advantage-estimator {algo["advantage_estimator"]}
        {_loss_args(algo)}
        {"" if algo["grpo_std_normalization"] else "--disable-grpo-std-normalization"}
        {"--calculate-per-token-loss" if algo["per_token_loss"] else ""}
        {"--use-rollout-routing-replay" if algo["routing_replay"] else ""}
        --kl-coef 0 --entropy-coef {algo["entropy_coef"]}
        {f"--use-kl-loss --kl-loss-coef {algo['kl_loss_coef']}" if kl_loss else ""}
        {f"--ref-load {model_dir(recipe)}" if kl_loss and backend(recipe) != "megatron" else ""}
        {f"--seed {algo['seed']}" if algo["seed"] is not None else ""}
        {"--normalize-advantages" if algo["advantage_estimator"].startswith("reinforce_plus_plus") else ""}
        --eps-clip {algo["eps_clip"]} --eps-clip-high {algo["eps_clip_high"]}
        --optimizer adam --lr {algo["lr"]} --lr-decay-style constant
        --weight-decay {algo["weight_decay"]} --adam-beta1 {beta1} --adam-beta2 {beta2}

        --custom-generate-function-path miles_ext.obs.loop_watchdog.generate
        --custom-agent-function-path miles_ext.harbor.agent_function.run
        --custom-agent-function-mode {recipe["agent"]["function_mode"]}
        --custom-rm-path miles_ext.harbor.reward.reward_func
        --dynamic-sampling-filter-path {GROUP_FILTERS[algo["group_filter"]]}
        --tito-model {preset.tito_model}
        --use-session-server --session-server-port {SESSION_PORT} --session-server-workers {SESSION_WORKERS}
        --no-namespaced-radix-cache
        --pin-rollout-manager-to-head

        --rollout-num-gpus-per-engine {recipe["rollout"]["gpus_per_engine"]}
        {_rollout_args(recipe)}
        --sglang-tool-call-parser {preset.tool_call_parser} --sglang-reasoning-parser {preset.reasoning_parser}
        --sglang-mem-fraction-static {mem_fraction}
        --sglang-decode-log-interval 1000
        --update-weight-buffer-size 536870912
        {_backend_args(recipe, resume)}
        {"--use-dynamic-batch-size" if preset.packed_batches else "--micro-batch-size 1"}
        --max-tokens-per-gpu {algo["max_tokens_per_gpu"]}
        {placement}

        --use-wandb --wandb-project {wandb_project(recipe, run_id)} --wandb-group {run_id}
        --custom-placement-hook-path miles_ext.obs.gpu_telemetry.start
        {f"--wandb-team {tracking['wandb_entity']}" if tracking["wandb_entity"] else ""}
        """
    )
    if preset.chat_template_kwargs:
        args += ["--apply-chat-template-kwargs", json.dumps(preset.chat_template_kwargs)]
    if recipe["checkpoint"]["stage_dir"]:
        args += ["--custom-megatron-save-context-path", "miles_ext.checkpoint.staged_save.context"]
    if recipe["checkpoint"]["save_rollout_data"]:
        args += ["--save-debug-rollout-data", str(RESULTS / run_id / "rollout_data" / "{rollout_id}.pt")]
    # torch_memory_saver (colocated offload) rejects expandable segments; a fully async trainer needs them so its
    # cache does not fragment between steps
    alloc = f"expandable_segments:{fully_async}"
    args += ["--train-env-vars", json.dumps({"PYTORCH_CUDA_ALLOC_CONF": alloc})]
    return args + [str(a) for a in recipe["extra_args"]]


def _loss_args(algo: dict) -> str:
    if algo["loss"] == "policy_loss":
        return f"--use-tis --tis-clip {algo['tis_clip']} --tis-clip-low {algo['tis_clip_low']}"
    return f"""--loss-type score_centering --use-rollout-logprobs
        --score-centering-is {algo["score_centering_is"]} --score-centering-tis-clip {algo["tis_clip"]}
        --rollout-top-logprobs-num {algo["top_logprobs"]}"""


def trainer_env(recipe: dict) -> dict[str, str]:
    """Env the Miles driver, trainer workers and engines need before Miles starts (Ray workers
    inherit the raylet's env): the model's, its rollout precision's, WANDB__DISABLE_STATS so that
    only the per-node observers of miles_ext.obs.gpu_telemetry sample system metrics, and for
    Megatron CUDA_DEVICE_MAX_CONNECTIONS=1, which its argument validation requires for TP or CP in
    every process that parses args."""
    preset = PRESETS[recipe["model"]["preset"]]
    env = {**preset.env, **preset.precisions[rollout_precision(recipe)], "WANDB__DISABLE_STATS": "true"}
    if backend(recipe) == "megatron":
        env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    if recipe["checkpoint"]["stage_dir"]:
        env[STAGE_ENV] = recipe["checkpoint"]["stage_dir"]
    return env


def train_dir(recipe: dict) -> Path:
    """The weights the trainer loads: the preset's BF16 copy when the engines serve a quantized
    checkpoint, else the engines' own."""
    preset = PRESETS[recipe["model"]["preset"]]
    if rollout_precision(recipe) == FP8_FP4:
        return MODELS / preset.train_dir_name
    return model_dir(recipe)


def _rollout_args(recipe: dict) -> str:
    preset, roll = PRESETS[recipe["model"]["preset"]], recipe["rollout"]
    args = preset.sglang_args.format(gpus=roll["gpus_per_engine"])
    if roll["context_length"]:
        # a reply that would overflow the context is shortened to fit instead of refused
        args += f" --sglang-context-length {roll['context_length']} --sglang-allow-auto-truncate"
    if roll["max_running_requests"]:
        args += f" --sglang-max-running-requests {roll['max_running_requests']}"
    if rollout_precision(recipe) == FP8_FP4:
        # ext-007: weight sync re-encodes each tensor in the format --hf-checkpoint stores it
        args += " --custom-weight-quantizer-path miles_ext.lowprec.checkpoint_quantizer.quantize_like_checkpoint"
    if roll["expert_qat"]:
        args += " --mxfp4-qat-routed-experts"  # fix-006
    return args


def _optimizer_args(recipe: dict, resume: bool) -> str:
    opt, ckpt = recipe["optimizer"], recipe["checkpoint"]
    args = ""
    if opt["offload"] == "disk":
        args += f" --stream-optimizer-state-to-disk --offload-train-disk-dir {opt['disk_dir']}"
        if opt["moment_dtype"]:
            args += f" --stream-optimizer-state-moment-dtype {opt['moment_dtype']}"
    if not ckpt["save_optimizer"]:
        args += " --no-save-optim" + (" --no-load-optim" if resume else "")
    return args


def _backend_args(recipe: dict, resume: bool) -> str:
    if backend(recipe) != "megatron":
        return "--gradient-checkpointing --attn-implementation flash_attention_2"
    # A fresh run loads from --ref-load when --load holds no Megatron checkpoint
    # (megatron_config.py); pointed at the HF weights, mbridge converts them on
    # load, so no torch_dist conversion. With kl_coef 0 no reference model is built.
    # The LR schedule's length is num_rollout x batch; on resume Megatron asserts it
    # matches the checkpoint's, so a run extended past its first num_rollout could
    # not resume. --override-opt_param-scheduler: this launch's schedule wins over
    # the checkpoint's (the recipes use a constant LR, so nothing else changes).
    par, preset = megatron_parallel(recipe), PRESETS[recipe["model"]["preset"]]
    return f"""
        {load_model_args(preset.megatron_model_type)}
        --ref-load {train_dir(recipe)}
        --tensor-model-parallel-size {par["tp"]} {"--sequence-parallel" if par["tp"] > 1 else ""}
        --pipeline-model-parallel-size {par["pp"]} --context-parallel-size {par["cp"]}
        {preset.megatron_cp_args if par["cp"] > 1 else ""}
        {f"--expert-model-parallel-size {par['ep']} --expert-tensor-parallel-size 1" if par["ep"] > 1 else ""}
        --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
        {f"--attention-backend {preset.attention_backend}" if preset.attention_backend else ""}
        --attention-softmax-in-fp32
        {_GRAD_DTYPE_FLAGS[recipe["optimizer"]["grad_dtype"]]}
        --attention-dropout 0.0 --hidden-dropout 0.0
        --log-probs-chunk-size 4096
        --override-opt_param-scheduler
        {preset.megatron_args}
        {_optimizer_args(recipe, resume)}
        {_lora_args(recipe)}
        """


def entrypoint(recipe: dict) -> str:
    return "train_async.py" if recipe["algorithm"]["mode"] == "fully_async" else "train.py"
