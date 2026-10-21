"""R1 (minimal): what a form or CLI needs to build a recipe without knowing Miles.

``schema()`` returns the recipe's JSON Schema (derived from ``recipe.DEFAULTS``),
the defaults, the presets, and the reserved flags: Miles flags that a recipe
key or the platform's safety layer sets, which ``extra_args`` may not pass.
The full spec model (apiVersion converters, field docs) is not built yet.
"""

import copy
import dataclasses

from miles_ext.recipes import recipe as R
from miles_ext.recipes.presets import PRESETS

API_VERSION = "miles-rl/v1"

# flag -> the recipe key or target field that owns it
RESERVED_FLAGS = {
    "--save": "checkpoint (path is fixed per run)",
    "--load": "checkpoint (resume is automatic)",
    "--ref-load": "model.preset / model.revision",
    "--hf-checkpoint": "model.preset / model.revision",
    "--save-hf": "checkpoint.save_hf",
    "--save-interval": "checkpoint.save_interval",
    "--save-trigger-sentinel": "platform (deadline save)",
    "--save-debug-rollout-data": "checkpoint.save_rollout_data",
    "--wandb-project": "tracking.wandb_project",
    "--wandb-group": "platform (run id)",
    "--wandb-team": "tracking.wandb_entity",
    "--actor-num-nodes": "cluster.train_gpus",
    "--actor-num-gpus-per-node": "cluster.train_gpus",
    "--rollout-num-gpus": "cluster.train_gpus",
    "--num-gpus-per-node": "cluster.gpu",
    "--colocate": "algorithm.mode",
    "--fully-async": "algorithm.mode",
    "--num-rollout": "algorithm.num_rollout",
    "--prompt-data": "taskset",
    "--custom-agent-function-path": "taskset.harness",
    "--custom-generate-function-path": "taskset.harness",
    "--custom-agent-function-mode": "agent.function_mode",
    "--session-server-port": "platform (session servers)",
    "--use-session-server": "platform (session servers)",
    # named knobs (r6): reserved even when the recipe leaves the flag out
    "--advantage-estimator": "algorithm.advantage_estimator",
    "--disable-grpo-std-normalization": "algorithm.grpo_std_normalization",
    "--calculate-per-token-loss": "algorithm.per_token_loss",
    "--loss-type": "algorithm.loss",
    "--use-tis": "algorithm.loss",
    "--use-rollout-logprobs": "algorithm.loss",
    "--score-centering-is": "algorithm.score_centering_is",
    "--score-centering-tis-clip": "algorithm.tis_clip",
    "--rollout-top-logprobs-num": "algorithm.top_logprobs",
    "--use-kl-loss": "algorithm.kl_loss_coef",
    "--kl-loss-coef": "algorithm.kl_loss_coef",
    "--entropy-coef": "algorithm.entropy_coef",
    "--seed": "algorithm.seed",
    "--max-weight-staleness": "algorithm.max_weight_staleness",
    "--over-sampling-batch-size": "async.over_sampling_batch_size",
    "--async-max-concurrent-samples": "async.max_concurrent_samples",
    "--async-unused-samples-handler": "async.unused_samples",
    "--pause-generation-mode": "async.pause_generation",
    "--async-single-pass": "async.single_pass",
    "--async-retry-aborted-tries": "async.retry_aborted_tries",
    "--async-checkpoint-buffer": "async.checkpoint_buffer",
    "--async-in-flight-controller-path": "async.in_flight (auto uses ext-035)",
    "--custom-async-data-buffer-path": "async.checkpoint_buffer (log uses the write-ahead group log, ext-034)",
    "--partial-rollout": "unsupported: Miles rejects it in fully async mode",
}


def emitted_flags(recipe: dict, run_id: str = "reserved-check") -> set[str]:
    """Every flag ``miles_args`` emits for this recipe (resume and fresh start), without its extra_args."""
    base = {**recipe, "extra_args": []}
    flags = set()
    for resume in (False, True):
        flags |= {a for a in R.miles_args(base, run_id, resume=resume) if a.startswith("--")}
    return flags


def reserved_flag_errors(recipe: dict, run_id: str = "reserved-check") -> list[dict]:
    """``extra_args`` may not repeat a flag the recipe already sets: the hand list names the key to
    use instead; any other flag ``miles_args`` emits for this recipe is reserved too (argparse would
    otherwise let the later copy win silently)."""
    try:
        emitted = emitted_flags(recipe, run_id)
    except (ValueError, KeyError, TypeError):
        emitted = set()  # the recipe is invalid; its own errors are reported
    errors = []
    for arg in map(str, recipe.get("extra_args") or []):
        flag = arg.split("=", 1)[0]
        if flag in RESERVED_FLAGS:
            owner = RESERVED_FLAGS[flag]
            message = f"{flag} is {owner}" if owner.startswith("unsupported") else f"{flag} is set by {owner}; use that"
            errors.append({"path": "extra_args", "message": message})
        elif flag in emitted:
            errors.append({"path": "extra_args", "message": f"{flag} is already set by the recipe"})
    return errors


def _schema_of(value, nullable: bool = False) -> dict:
    if isinstance(value, dict):
        return {"type": "object", "additionalProperties": False,
                "properties": {k: _schema_of(v, v is None) for k, v in value.items()}}
    if isinstance(value, bool):
        t = {"type": "boolean"}
    elif isinstance(value, int):
        t = {"type": "integer"}
    elif isinstance(value, float):
        t = {"type": "number"}
    elif isinstance(value, list):
        t = {"type": "array"}
    elif isinstance(value, str):
        t = {"type": "string"}
    else:
        return {}  # None default: any type (the recipe validator decides)
    return {"anyOf": [t, {"type": "null"}]} if nullable else t


_JSON_TYPES = {"int": "integer", "number": "number", "str": "string", "bool": "boolean", "list": "array"}


def _rule_schema(rule: dict) -> dict:
    t = {"type": _JSON_TYPES[rule["type"]]}
    if "enum" in rule:
        t["enum"] = list(rule["enum"])
    if "min" in rule:
        t["exclusiveMinimum" if rule.get("exclusive_min") else "minimum"] = rule["min"]
    if "max" in rule:
        t["maximum"] = rule["max"]
    return {"anyOf": [t, {"type": "null"}]} if rule.get("nullable") else t


def schema() -> dict:
    json_schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "Miles recipe",
                   **_schema_of(R.DEFAULTS)}
    for path, rule in R.FIELD_RULES.items():
        node = json_schema
        *parents, leaf = path.split(".")
        for key in parents:
            node = node["properties"][key]
        node["properties"][leaf] = _rule_schema(rule)
    return {
        "api_version": API_VERSION,
        "json_schema": json_schema,
        "defaults": copy.deepcopy(R.DEFAULTS),
        "presets": {k: dataclasses.asdict(v) for k, v in PRESETS.items()},
        "reserved_flags": copy.deepcopy(RESERVED_FLAGS),
        "namespaces": list(R.OWN_APP_NAMESPACES),
    }
