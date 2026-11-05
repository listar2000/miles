"""``MILES_STATE=modal-dict``: run state in the caller's progress Dict, which ``miles_ext.platform`` reads."""

import json
import os

import modal

DICT_ENV = "MILES_MODAL_DICT"  # {"name": ..., "environment": ...}, set by the caller


def modal_dict() -> modal.Dict:
    spec = json.loads(os.environ[DICT_ENV])
    return modal.Dict.from_name(spec["name"], environment_name=spec["environment"], create_if_missing=True)
