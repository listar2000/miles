"""ext-038: single-turn tasks (taskset.harness single-turn)."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
import yaml

from miles_ext.recipes import recipe as R
from miles_ext.tasks import single_turn

ROOT = Path(__file__).resolve().parents[2]
EXACT = f"{__name__}.exact_match"


def exact_match(reply: str, label: str) -> tuple[int, str]:
    return (1, "ok") if reply == label else (0, "wrong")


def _recipe(**taskset) -> dict:
    spec = yaml.safe_load((ROOT / "miles_ext/recipes/configs/dsv41-flash-tb4-overfit16.yaml").read_text())
    spec["taskset"] = {"harness": R.HARNESS_SINGLE_TURN, "prompts": "/host/prompts.jsonl", **taskset}
    return R.from_dict(spec)


def test_recipe_single_turn_args_and_env():
    r = _recipe()
    R.validate(r)
    assert R.is_single_turn(r) and not R.reads_task_dirs(r) and not R.needs_tunnel(r)
    assert R.harbor_env(r, "run1") == {"MILES_EXT_HARNESS": R.HARNESS_SINGLE_TURN}
    argv = R.miles_args(r, "run1", resume=False)
    assert argv[argv.index("--custom-agent-function-path") + 1] == "miles_ext.tasks.single_turn.run"
    assert argv[argv.index("--custom-rm-path") + 1] == "miles_ext.harbor.reward.reward_func"


def test_recipe_agent_harness_unchanged():
    spec = yaml.safe_load((ROOT / "miles_ext/recipes/configs/dsv41-flash-tb4-overfit16.yaml").read_text())
    r = R.from_dict(spec)
    argv = R.miles_args(r, "run1", resume=False)
    assert argv[argv.index("--custom-agent-function-path") + 1] == "miles_ext.harbor.agent_function.run"
    assert "HARBOR_TASKS_DIR" in R.harbor_env(r, "run1")


@pytest.mark.parametrize(
    "taskset, message",
    [
        ({"tasks_dir": "/host/tasks"}, "no task dirs"),
        ({"tasks_volume": "miles-datasets:/x"}, "no task dirs"),
        ({"prompts": None}, "taskset.prompts is required"),
    ],
)
def test_recipe_single_turn_refuses(taskset, message):
    with pytest.raises(ValueError, match=message):
        R.validate(_recipe(**taskset))


@pytest.mark.parametrize(
    "metadata, message",
    [
        ({"label": "1"}, "metadata.task"),
        ({"task": "nope", "label": "1"}, "metadata.task"),
        ({"task": "stackvm", "label": "1"}, "metadata.task"),
        ({"task": f"{__name__}.missing", "label": "1"}, "metadata.task"),
        ({"task": f"{__name__}.EXACT", "label": "1"}, "metadata.task"),
        ({"task": EXACT}, "label"),
    ],
)
def test_check_row_refuses(metadata, message):
    with pytest.raises(ValueError, match=message):
        single_turn.check_row({"metadata": metadata})


def test_agent_function_grades_one_reply(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        msg = {"role": "assistant", "content": "12 0 5", "reasoning_content": "work"}
        return httpx.Response(200, json={"choices": [{"message": msg, "finish_reason": "stop"}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(
        single_turn.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )
    out = asyncio.run(
        single_turn.run(
            "http://h:1/sessions/s",
            "run it",
            {"temperature": 1.0, "max_tokens": 64},
            {"task": EXACT, "label": "12 0 5"},
        )
    )
    assert out == {"reward": 1.0, "grade": "ok", "finish_reason": "stop"}
    assert seen["url"] == "http://h:1/sessions/s/v1/chat/completions"
    assert seen["body"]["messages"] == [{"role": "user", "content": "run it"}]
    assert seen["body"]["temperature"] == 1.0 and seen["body"]["stream"] is False
