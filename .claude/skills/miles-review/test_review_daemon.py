"""Trigger rules of the review daemon: python -m pytest .claude/skills/miles-review"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))

import review_daemon as D  # noqa: E402

ME = {"listar2000"}


def _pr(draft=False, files=("miles_ext/x.py",), labels=(), author="listar2000", head="h1"):
    return {
        "number": 7,
        "headRefOid": head,
        "isDraft": draft,
        "author": {"login": author},
        "isCrossRepository": False,
        "labels": [{"name": n} for n in labels],
        "files": [{"path": f} for f in files],
        "body": "",
    }


def _c(body, user="listar2000", id=1):
    return {"id": id, "user": {"login": user}, "body": body}


def _review(head, mode="normal"):
    return _c(f"<!-- miles-review head={head} mode={mode} verdict=ready -->\nreport")


def _decide(pr, thread=(), done=(), settled=True):
    return D.decide(
        pr, list(thread), ME, set(done), 1000.0, {} if settled else {pr["headRefOid"]: 999.0}, 0 if settled else 60
    )


def test_ready_pr_is_reviewed_once_per_head():
    assert _decide(_pr()) == ("normal", None)
    assert _decide(_pr(), [_review("h1")]) is None
    assert _decide(_pr(head="h2"), [_review("h1")]) == ("normal", None)


def test_drafts_and_unsettled_heads_wait_but_a_command_runs_at_once():
    command = _c("/review", id=9)
    assert _decide(_pr(draft=True)) is None
    assert _decide(_pr(), settled=False) is None
    assert _decide(_pr(draft=True), [_review("h1"), command]) == ("normal", command)
    assert _decide(_pr(draft=True), [command, _review("h1")]) is None  # already answered
    assert _decide(_pr(draft=True), [_review("h1"), command], done=["cmd:9"]) is None  # answered in a dry run


def test_deep_from_command_label_or_trainer_paths():
    assert _decide(_pr(), [_c("/review deep")])[0] == "deep"
    assert _decide(_pr(labels=["deep-review"])) == ("deep", None)
    assert _decide(_pr(files=["miles/rollout/x.py"])) == ("deep", None)
    assert _decide(_pr(labels=["deep-review"]), [_review("h1", "normal")]) == ("deep", None)


def test_only_allowed_users_trigger_or_mark_reviews():
    assert _decide(_pr(author="someone")) is None
    assert _decide(_pr(draft=True), [_c("/review", user="someone")]) is None
    assert _decide(_pr(), [_c(_review("h1")["body"], user="someone")]) == ("normal", None)


def test_merge_needs_a_ready_review_at_the_head_and_answers_once():
    pr, ask = {**_pr(), "baseRefName": "main"}, _c("/merge", id=5)
    assert D.merge_request([_review("h1"), ask], ME, set()) == ask
    assert D.merge_request([ask, _c("<!-- miles-merge result=refused -->\nNot merged")], ME, set()) is None
    assert D.merge_request([_c("/merge", user="someone")], ME, set()) is None
    assert D.merge_refusal(pr, [_review("h1")], ME) is None
    assert "no Codex review" in D.merge_refusal({**pr, "headRefOid": "h2"}, [_review("h1")], ME)
    assert "stacked" in D.merge_refusal({**pr, "baseRefName": "topic"}, [_review("h1")], ME)
    blocked = _c("<!-- miles-review head=h1 mode=normal verdict=blocked -->")
    assert "blocked" in D.merge_refusal(pr, [_review("h1"), blocked], ME)
    assert "deep" in D.merge_refusal({**pr, "files": [{"path": "miles/x.py"}]}, [_review("h1")], ME)


def test_pr_body_lines_cannot_add_review_options():
    pr = {
        **_pr(),
        "body": "Review context: fork/README.md --checks-on local\nEvidence: /root/.codex/auth.json modal://miles-x/a",
    }
    args = SimpleNamespace(python="py", out="o", checks_on="modal", repo="r", dry_run=True, board=None)
    argv = D.review_argv(args, pr, "deep", None)
    assert [a for a in argv if a.startswith("-")].count("--checks-on") == 1 and "/root/.codex/auth.json" not in argv
    assert argv[-5:] == ["--context", "fork/README.md", "local", "--evidence", "modal://miles-x/a"]
