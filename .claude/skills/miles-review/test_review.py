"""How a review round is rendered and anchored: python -m pytest .claude/skills/miles-review"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import review as R  # noqa: E402

META = {"number": 7, "headRefOid": "b" * 40, "folded": None}
RUN = {"agent": "codex", "model": "m", "effort": "high", "protocol": "p", "seconds": 1, "attempt": 1}
CHECKS = {k: "ledger check: ok" for k in ("ledger", "precommit", "pytest")}
CTX = {"files": ["a.py"], "proofs": {}, "checks": {"head": CHECKS, "base": CHECKS}}


def _finding(location, line, exact=False):
    return {"severity": "should-fix", "rule": "comment", "title": f"t-{location}", "location": location, "line": line,
            "end_line": line, "evidence": "e", "suggestion": "s", "exact": exact}  # fmt: skip


def test_round_history_and_line_findings_only_listed():
    r = {"verdict": "ready after the should-fixes", "summary": "x", "handoffs": [],
         "findings": [_finding("a.py", 3, exact=True), _finding("PR body", 0)],
         "prior": [{"title": "old", "status": "fixed"}]}  # fmt: skip
    prior = "<!-- miles-review head=aaa mode=normal verdict=blocked round=2 -->\n<!-- this-round: Round 2 at `aaa`: blocked -->"
    rnd, text = R.render(r, META, "c" * 40, RUN, [], CTX, prior, inline=[0])
    assert rnd == 3 and "- Round 2 at `aaa`: blocked" in text and "Fixed since round 2: 1 (old)" in text
    assert "(comment on the line)" in text and "```suggestion" not in text  # the line comment carries it
    assert "Suggested (your call): s" in text
    assert "```suggestion\ns\n```" in R.line_comment(r["findings"][0])["body"]


def test_diff_lines_are_the_head_side_of_each_hunk(monkeypatch):
    patch = "@@ -1,3 +1,4 @@\n a\n-b\n+c\n+d\n e\n@@ -10,1 +11,1 @@\n-x\n+y"
    monkeypatch.setattr(R, "gh_api", lambda *a: [[{"filename": "a.py", "patch": patch}]])
    assert R.diff_lines(7) == {"a.py": {1, 2, 3, 4, 11}}
