"""How a review round is rendered and anchored: python -m pytest .claude/skills/miles-review"""

import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_regressions_count_failures_not_wording():
    assert R.regressed("pytest", "1 failed, 434 passed", "435 passed")
    assert not R.regressed("pytest", "435 passed, 1 error", "437 passed, 1 error")
    assert R.regressed("precommit", "failed: isort, Black", "failed: isort")
    assert not R.regressed("ledger", "ledger check: ok", "ledger check: ok")


def test_marker_round_trips_the_session_for_both_readers():
    import review_daemon as D

    m = R.MARKER.format(head="a" * 40, mode="deep", verdict="ready", round=2, session="01a1-9f", rules="c" * 12)
    assert R.ROUND.search(m).groupdict() == {"head": "a" * 40, "mode": "deep", "round": "2", "session": "01a1-9f",
                                             "rules": "c" * 12}  # fmt: skip
    assert D.MARKER.match(m) and D.MARKER.match("<!-- miles-review head=aa mode=normal verdict=ready -->")


def test_follow_up_resumes_only_on_the_same_base_rules_and_mode(tmp_path):
    def git(*a):
        return R.sh(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=tmp_path).stdout.strip()

    git("init", "-q")
    for name in ("base", "prior", "head"):
        (tmp_path / "f.py").write_text(f"{name}\n")
        git("add", "f.py")
        git("commit", "-qm", name)
    revs = git("log", "--format=%H").split()[::-1]
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "pytest.txt").write_text("435 passed")
    args = SimpleNamespace(fresh=False, deep=False, pr=7, repo=tmp_path)
    last = {"head": revs[1], "mode": "normal", "round": "2", "session": "s1", "rules": "r1"}
    meta = {"headRefOid": revs[2], "rules": "r1"}
    prompt = R.follow_up(args, tmp_path, meta, revs[0], bundle, last)
    assert "Round 3 of your review" in prompt and "-prior\n+head" in prompt and "435 passed" in prompt
    assert R.follow_up(args, tmp_path, meta, revs[2], bundle, last) is None  # rebased: base not under the old head
    assert R.follow_up(args, tmp_path, meta, revs[0], bundle, {**last, "rules": "old"}) is None
    assert R.follow_up(args, tmp_path, meta, revs[0], bundle, {**last, "mode": "deep"}) is None
    assert R.follow_up(SimpleNamespace(**{**vars(args), "fresh": True}), tmp_path, meta, revs[0], bundle, last) is None


def test_new_tests_run_in_one_pytest_with_a_result_each():
    calls = []

    def run(cmd, timeout=None, check=False):
        calls.append(cmd)
        out = "PASSED t.py::test_a\nFAILED t.py::test_b[1] - assert 0\nPASSED t.py::test_b[2]\n"
        return R.subprocess.CompletedProcess(cmd, 1, out, "")

    got = R.pytest_nodes(SimpleNamespace(python="python", run=run), ["t.py::test_a", "t.py::test_b", "t.py::test_c"])
    assert len(calls) == 1 and {n: r[0] for n, r in got.items()} == {
        "t.py::test_a": "pass", "t.py::test_b": "fail", "t.py::test_c": "not run"}  # fmt: skip
