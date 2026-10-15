"""R1 (minimal): what a form or CLI needs to build a recipe without knowing Miles.

``schema()`` returns the recipe's JSON Schema (derived from ``recipe.DEFAULTS``),
the defaults, the presets, and the reserved flags: Miles flags that a recipe
key or the platform's safety layer sets, which ``extra_args`` may not pass.
The full spec model (apiVersion converters, field docs) comes with r4.
"""

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
    "--session-server-port": "platform (session servers)",
    "--use-session-server": "platform (session servers)",
}


def reserved_flag_errors(recipe: dict) -> list[dict]:
    errors = []
    for arg in map(str, recipe.get("extra_args") or []):
        flag = arg.split("=", 1)[0]
        if flag in RESERVED_FLAGS:
            errors.append({"path": "extra_args", "message": f"{flag} is set by {RESERVED_FLAGS[flag]}; use that"})
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


def schema() -> dict:
    json_schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "Miles recipe",
                   **_schema_of(R.DEFAULTS)}
    json_schema["properties"]["model"]["properties"]["preset"] = {"enum": sorted(PRESETS)}
    json_schema["properties"]["sandbox"]["properties"]["provider"] = {"enum": list(R.SANDBOX_PROVIDERS)}
    json_schema["properties"]["model"]["properties"]["backend"] = {"enum": [*R.BACKENDS, None]}
    return {
        "api_version": API_VERSION,
        "json_schema": json_schema,
        "defaults": R.DEFAULTS,
        "presets": {k: dataclasses.asdict(v) for k, v in PRESETS.items()},
        "reserved_flags": RESERVED_FLAGS,
        "namespaces": list(R.OWN_APP_NAMESPACES),
    }
