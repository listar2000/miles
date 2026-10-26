"""Grade a STAK reply against the expected final stack."""

import re

TASK = "stackvm"
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)


def grade(response: str, label: str) -> tuple[int, str]:
    """(reward, kind): kind is ok, wrong, no_answer, multi_answer or bad_format. Only text after the reasoning counts."""
    text = response.split("</think>")[-1]
    found = ANSWER.findall(text)
    if not found:
        return 0, "no_answer"
    if len(found) > 1:
        return 0, "multi_answer"
    got = found[0].strip().split()
    if not got or not all(re.fullmatch(r"\d+", t) for t in got):
        return 0, "bad_format"
    return (1, "ok") if " ".join(str(int(t)) for t in got) == label else (0, "wrong")
