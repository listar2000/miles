"""The DAPO math grader."""

import pytest

from miles_ext.tasks import dapo_math


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("x\nAnswer: \\boxed{34}", (1, "ok")),
        ("x\nAnswer: \\boxed{35}", (0, "wrong")),
        ("no final line", (0, "no_answer")),
    ],
)
def test_grade(reply, expected):
    assert dapo_math.grade(reply, "34") == expected
