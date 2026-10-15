"""The fork's platform API: what the platform's controller calls, with a ``target`` last.

Importable without GPU dependencies (needs ``modal`` and the fork's ``miles`` tree).
In r3: ``schema``, ``validate``, ``teardown``/``resource_names``, the per-run
secret, ``prepare_weights``/``list_weights``/``delete_weights``, and
``list_checkpoints``/``delete_checkpoint``/``apply_retention``/``finalize_hf``.
``launch`` and ``observe`` (with the rank-0 supervisor and exit record) come in r4;
until then runs start through ``miles_ext.launch.modal_app`` with the same target
values in env (MILES_APP_NAME, MILES_RESULTS_VOLUME, MILES_WEIGHTS_VOLUME,
MILES_SECRET_NAME). Every change here is caller-facing: say so in the release note.
"""

from miles_ext.platform.checkpoints import (apply_retention, delete_checkpoint, finalize_hf, list_checkpoints,
                                            retention_plan)
from miles_ext.platform.schema import API_VERSION, RESERVED_FLAGS, schema
from miles_ext.platform.secrets import create_run_secret, delete_run_secret, required_keys
from miles_ext.platform.target import Target, run_secret_name
from miles_ext.platform.teardown import resource_names, teardown
from miles_ext.platform.validate import instant_errors, validate
from miles_ext.platform.weights import delete_weights, list_weights, prepare_weights, weights_key

__all__ = [
    "API_VERSION", "RESERVED_FLAGS", "Target", "apply_retention", "create_run_secret", "delete_checkpoint",
    "delete_run_secret", "delete_weights", "finalize_hf", "instant_errors", "list_checkpoints", "list_weights",
    "prepare_weights", "required_keys", "resource_names", "retention_plan", "run_secret_name", "schema",
    "teardown", "validate", "weights_key",
]
