"""The grader's verdicts hold and every prompt row grades its own label."""

import json
from pathlib import Path

import pytest

from experiments.dsv41_stak_overfit import stackvm
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
        ("<answer>12 0 5 0</answer>", (0, "wrong")),
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


def test_every_prompt_row_grades_its_own_label():
    rows = [json.loads(line) for line in (HERE / "prompts/stak32.jsonl").read_text().splitlines()]
    assert len(rows) == 32
    for row in rows:
        single_turn.check_row(row)
        label = row["metadata"]["label"]
        assert stackvm.grade(f"<answer>{label}</answer>", label) == (1, "ok")
