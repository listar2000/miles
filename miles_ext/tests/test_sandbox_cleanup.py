"""Sandbox cleanup never reaches another team's apps.

The Modal and Sail workspaces are shared (other teams run e.g. ``mobius-<uuid>-N``
apps there). Exit cleanup terminates every sandbox in the apps it names, so it
must name only this run's apps, exactly. CPU only, no network:

    python -m pytest miles_ext/tests/test_sandbox_cleanup.py
"""

import ast
import copy
import inspect
import types
from pathlib import Path

import pytest

from miles_ext.recipes import recipe as R

ROOT = Path(inspect.getfile(R)).parents[2]

OTHERS = ["mobius-26d3eb30-007c-48ef-a700-805f1926f786-1", "mobius-local-regrade-b00524e1-2971816-1",
          "mobius-honeycomb-7fe29c65", "burr-harbor-validation", "__harbor__", "harbor-official-digest-rest42"]


def _loaded(prefix: str, monkeypatch) -> dict:
    monkeypatch.setenv("MILES_DATA", "/data")
    recipe = R.load(str(Path(inspect.getfile(R)).parent / "configs" / "qwen35-9b-swegym8-sail.yaml"))
    recipe["sandbox"]["app_prefix"] = prefix
    return recipe


def _recipe(prefix: str = R.DEFAULT_APP_PREFIX) -> dict:
    recipe = copy.deepcopy(R.DEFAULTS)
    recipe["sandbox"]["app_prefix"] = prefix
    return recipe


def test_cleanup_names_are_exactly_this_runs_apps():
    assert R.sandbox_app_names_for_cleanup(_recipe(), "q35-sail-10step-v1") == [
        "miles-mobius-q35-sail-10step-v1", "miles-fork-sbx-q35-sail-10step-v1"]
    # a custom prefix gets only its own name, no legacy names
    assert R.sandbox_app_names_for_cleanup(_recipe("mdev-"), "job-7") == ["mdev-job-7"]


@pytest.mark.parametrize("prefix", ["mobius-", "harbor-", "burr-", "x-"])
def test_foreign_prefix_rejected(prefix, monkeypatch):
    R.validate(_loaded(R.DEFAULT_APP_PREFIX, monkeypatch))
    with pytest.raises(ValueError, match="app_prefix must start with"):
        R.validate(_loaded(prefix, monkeypatch))
    with pytest.raises(ValueError):
        R.sandbox_app_name(_recipe(prefix), "26d3eb30-1")


@pytest.mark.parametrize("run_id", ["", "*", "a*", "?", "[a-z]", "a b", "../x", "a/b", "-x"])
def test_run_id_cannot_widen_the_name(run_id):
    with pytest.raises(ValueError):
        R.sandbox_app_names_for_cleanup(_recipe(), run_id)


@pytest.mark.parametrize("name", OTHERS)
def test_other_teams_names_refused(name):
    with pytest.raises(ValueError):
        R.check_own_app_name(name)


def test_sail_cleanup_matches_exact_name_only():
    terminated = []

    class Box:
        def __init__(self, app):
            self.app, self.status = app, "running"

        def terminate(self):
            terminated.append(self.app)

    names = ["miles-mobius-run1", "miles-mobius-run10", "miles-mobius-run1-b", *OTHERS]
    apps = [types.SimpleNamespace(name=n) for n in names]
    stubs = {"App": types.SimpleNamespace(list=lambda: apps),
             "Sailbox": types.SimpleNamespace(list=lambda app_id: [Box(app_id.name), Box(app_id.name)])}
    # the module imports the vendored provider (needs Harbor); run the function on its own
    namespace = dict(stubs)
    exec(_function_source(str(ROOT / "miles_ext/harbor/sail/__init__.py"), "terminate_app_sailboxes"), namespace)
    terminate_app_sailboxes = namespace["terminate_app_sailboxes"]

    assert terminate_app_sailboxes("miles-mobius-run1") == 2
    assert terminated == ["miles-mobius-run1"] * 2
    for name in OTHERS:
        with pytest.raises(ValueError):
            terminate_app_sailboxes(name)
    assert terminated == ["miles-mobius-run1"] * 2


def _function_source(path: str, name: str) -> str:
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(Path(path).read_text(), node)


@pytest.mark.parametrize("path,name", [
    ("miles_ext/launch/modal_app.py", "_terminate_task_sandboxes"),
    ("miles_ext/launch/modal_app.py", "_terminate_app_sandboxes"),
    ("miles_ext/harbor/sail/__init__.py", "terminate_app_sailboxes"),
])
def test_cleanup_code_has_no_pattern_matching(path, name):
    src = _function_source(str(ROOT / path), name)
    for banned in ("startswith", "fnmatch", "glob", "re.match", "re.search", " in a.name", "find("):
        assert banned not in src, f"{name} uses {banned!r}; cleanup must match exact app names"
    assert "check_own_app_name" in src or "sandbox_app_names_for_cleanup" in src

