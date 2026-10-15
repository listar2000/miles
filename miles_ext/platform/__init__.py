"""The fork's platform API: what the platform's controller calls, with a ``target`` last.

Importable without GPU dependencies (needs ``modal`` and the fork's ``miles`` tree).
Since r3: ``schema``, ``validate``, ``teardown``/``resource_names``, the per-run
secret, ``prepare_weights``/``list_weights``/``delete_weights``, and
``list_checkpoints``/``delete_checkpoint``/``apply_retention``/``finalize_hf``.
Since r5: ``launch``, ``observe``/``completion``/``renew_lease`` (the rank-0
supervisor and the exit record are in the trainer). The test launcher
(``miles_ext.launch.modal_app``) still reads the same target values from env.
Every change here is caller-facing: say so in the release note.
"""

from miles_ext.platform.checkpoints import (apply_retention, delete_checkpoint, finalize_hf, list_checkpoints,
                                            retention_plan)
from miles_ext.platform.launch import launch
from miles_ext.platform.progress import completion, next_rollout_id, observe, renew_lease
from miles_ext.platform.schema import API_VERSION, RESERVED_FLAGS, schema
from miles_ext.platform.secrets import create_run_secret, delete_run_secret, required_keys
from miles_ext.platform.target import Target, run_secret_name
from miles_ext.platform.teardown import resource_names, teardown
from miles_ext.platform.validate import instant_errors, validate
from miles_ext.platform.weights import delete_weights, list_weights, prepare_weights, weights_key

__all__ = [
    "API_VERSION", "RESERVED_FLAGS", "Target", "apply_retention", "completion", "create_run_secret",
    "delete_checkpoint", "launch", "next_rollout_id", "observe", "renew_lease",
    "delete_run_secret", "delete_weights", "finalize_hf", "instant_errors", "list_checkpoints", "list_weights",
    "prepare_weights", "required_keys", "resource_names", "retention_plan", "run_secret_name", "schema",
    "teardown", "validate", "weights_key",
]
