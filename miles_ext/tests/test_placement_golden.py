"""Every recipe in configs/ keeps the Miles argv and the Modal cluster shape recorded in golden/."""

import json
import os
from pathlib import Path

import pytest

from miles_ext.recipes import recipe as R

CONFIGS = Path(__file__).resolve().parents[1] / "recipes" / "configs"
GOLDEN = Path(__file__).resolve().parent / "golden"
RECIPES = sorted(p.stem for p in CONFIGS.glob("*.yaml"))


def _record(name: str, monkeypatch) -> dict:
    monkeypatch.setenv("MILES_DATA", "/data")  # tasks_dir and prompts paths only; the argv does not hold them
    recipe = R.load(CONFIGS / f"{name}.yaml")
    return {
        "shape": R.shape(recipe),
        "entrypoint": R.entrypoint(recipe),
        "argv": R.miles_args(recipe, "run1", resume=False),
        "argv_resume": R.miles_args(recipe, "run1", resume=True),
    }


@pytest.mark.parametrize("name", RECIPES)
def test_recipe_launch_matches_golden(name, monkeypatch):
    record, path = _record(name, monkeypatch), GOLDEN / f"{name}.json"
    if os.environ.get("MILES_GOLDEN_WRITE") == "1":
        path.write_text(json.dumps(record, indent=1) + "\n")  # regenerate on purpose, then rerun without it
        pytest.skip(f"wrote {path.name}")
    assert record == json.loads(path.read_text())
