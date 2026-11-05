"""Both recipes resolve as the experiment needs, every prompt row grades, and the grader's verdicts hold."""

import json
from pathlib import Path

import pytest

from experiments.dsv41_stak_overfit import stackvm
from miles_ext.recipes import recipe as R
from miles_ext.tasks import single_turn

HERE = Path(__file__).parent


@pytest.mark.parametrize(
    "reply, verdict",
    [
        ("<think>maybe <answer>1</answer></think>\nSo <answer>12 0 5</answer>", (1, "ok")),  # reasoning ignored
        ("<answer>12 00 5</answer>", (1, "ok")),
        ("<answer> 12  0\n5 </answer>", (1, "ok")),
        ("<answer>12 0</answer>", (0, "wrong")),
        ("<answer>5 0 12</answer>", (0, "wrong")),
        ("12 0 5", (0, "no_answer")),
        ("<think>cut off before answering <answer>12 0 5", (0, "no_answer")),
        ("<answer>12 0 5</answer> or <answer>5 0 12</answer>", (0, "multi_answer")),
        ("<answer>12, 0, 5</answer>", (0, "bad_format")),
        ("<answer>-3</answer>", (0, "bad_format")),
        ("<answer></answer>", (0, "bad_format")),
    ],
)
def test_grade(reply, verdict):
    assert stackvm.grade(reply, "12 0 5") == verdict


def test_generator_selftest():
    stackvm.selftest()


def test_every_prompt_row_grades_its_own_label():
    rows = [json.loads(line) for line in (HERE / "prompts/stak32.jsonl").read_text().splitlines()]
    assert len(rows) == 32
    for row in rows:
        single_turn.check_row(row)
        label = row["metadata"]["label"]
        assert stackvm.grade(f"<answer>{label}</answer>", label) == (1, "ok")


@pytest.mark.parametrize("name", ["stak-nosc.yaml", "stak-sc.yaml"])
def test_recipe_argv(name):
    argv = R.miles_args(R.load(HERE / name), "run1", resume=False)
    assert argv[argv.index("--lora-rank") + 1] == "32"
    assert argv[argv.index("--rollout-top-p") + 1] == "0.95"
    assert argv[argv.index("--sglang-speculative-algorithm") + 1] == "DSPARK"
    assert argv[len(argv) - 1 - argv[::-1].index("--apply-chat-template-kwargs") + 1] == '{"thinking": false}'
    assert ("--loss-type" in argv) == (name == "stak-sc.yaml")
