"""taskset.tasks_volume: the task dirs on a Modal volume instead of the launching machine."""

import json
from pathlib import Path

import pytest
import yaml

from miles_ext.platform.validate import _inputs, _task_dirs_for_prompts
from miles_ext.recipes import recipe as R

ROOT = Path(__file__).resolve().parents[2]


def _recipe(**taskset) -> dict:
    spec = yaml.safe_load((ROOT / "miles_ext/recipes/configs/dsv41-flash-tb4-overfit16.yaml").read_text())
    spec["taskset"] = {"harness": spec["taskset"]["harness"], "prompts": "/host/prompts.jsonl", **taskset}
    return R.from_dict(spec)


def test_tasks_volume_mounts_at_datasets():
    r = _recipe(tasks_volume="miles-datasets:/tb4/v1/tasks")
    R.validate(r)
    assert R.tasks_volume(r) == ("miles-datasets", "/tb4/v1/tasks")
    assert R.tasks_path(r) == Path("/datasets/tb4/v1/tasks")
    assert R.harbor_env(r, "run1")["HARBOR_TASKS_DIR"] == "/datasets/tb4/v1/tasks"
    # the prompt file stays where Miles reads it
    argv = R.miles_args(r, "run1", resume=False)
    assert argv[argv.index("--prompt-data") + 1] == str(R.PROMPTS)


def test_tasks_dir_unchanged():
    r = _recipe(tasks_dir="/host/tasks")
    R.validate(r)
    assert R.tasks_volume(r) is None
    assert R.harbor_env(r, "run1")["HARBOR_TASKS_DIR"] == str(R.TASKS)


@pytest.mark.parametrize(
    "taskset, message",
    [
        ({"tasks_dir": "/host/tasks", "tasks_volume": "miles-datasets:/tb4"}, "not both"),
        ({}, "one of taskset.tasks_dir, taskset.tasks_volume"),
        ({"tasks_volume": "miles-datasets"}, "<volume>:/<path>"),
        ({"tasks_volume": "miles-datasets:tb4"}, "<volume>:/<path>"),
        ({"tasks_volume": "miles-datasets:/tb4/../x"}, "<volume>:/<path>"),
        ({"tasks_volume": "__harbor__:/tasks"}, "must start with one of"),
    ],
)
def test_tasks_volume_refuses(taskset, message):
    with pytest.raises(ValueError, match=message):
        R.validate(_recipe(**taskset))


def test_platform_spec_without_local_inputs_still_validates():
    r = _recipe()
    R.validate(r, local_inputs=False)
    assert R.tasks_path(r) == R.TASKS


def _taskset(tmp_path, ids, rows) -> tuple[Path, Path]:
    for i in ids:
        (tmp_path / "tasks" / i).mkdir(parents=True)
    (tmp_path / "prompts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path / "tasks", tmp_path / "prompts.jsonl"


def test_prompt_ids_need_task_dirs(tmp_path):
    tasks, prompts = _taskset(
        tmp_path, ["a", "b"], [{"metadata": {"instance_id": "a"}}, {"metadata": {"instance_id": "c"}}]
    )
    with pytest.raises(ValueError, match=r"1 prompt ids have no task dir.*\['c'\]"):
        _task_dirs_for_prompts(tasks, prompts)


def test_prompt_row_without_instance_id(tmp_path):
    tasks, prompts = _taskset(tmp_path, ["a"], [{"metadata": {"instance_id": "a"}}, {"prompt": "x"}])
    with pytest.raises(ValueError, match="prompt row 2 has no metadata.instance_id"):
        _task_dirs_for_prompts(tasks, prompts)


def test_prompt_ids_all_present(tmp_path):
    tasks, prompts = _taskset(tmp_path, ["a", "b"], [{"metadata": {"instance_id": "b"}}])
    assert _task_dirs_for_prompts(tasks, prompts) == (["a", "b"], ["b"])


def test_volume_inputs_without_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "TASKS", tmp_path / "inputs/tasks")
    monkeypatch.setattr(R, "RESULTS", tmp_path / "results")
    src = tmp_path / "results/run1/inputs"
    src.mkdir(parents=True)
    (src / R.PROMPTS.name).write_text("")
    with pytest.raises(FileNotFoundError):
        _inputs("run1")  # a tasks_dir run needs tasks/ there
    _inputs("run1", tasks=False)
    assert (tmp_path / "inputs").resolve() == src.resolve()
