"""Test the ATIF connector (RFC 0003) on Harbor's own trial directories and on small ones written here."""

import copy
import json
from pathlib import Path

import pytest

from miles_ext.sft.connectors import atif
from miles_ext.sft.connectors.atif import AtifOptions, convert_trial, find_trials
from miles_ext.sft.connectors.base import Dropped, report_path, write_records
from miles_ext.sft.record import parse_record

HARBOR = Path(__file__).parent / "fixtures" / "atif"


def _convert(trial, **options):
    items = list(convert_trial(trial, AtifOptions(**options)))
    records = [item for item in items if not isinstance(item, Dropped)]
    dropped = [item for item in items if isinstance(item, Dropped)]
    for record in records:
        parse_record(record)
    return records, dropped


def _by_trajectory(records):
    return {record["metadata"]["trajectory"]: record for record in records}


def _steps(record):
    return [(list(s.context), s.completion, s.train) for s in parse_record(record).steps]


def _roles(record):
    return [m["role"] for m in record["messages"]]


# ---- a trial directory written by the tests ----

TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}]


def _doc(steps, *, model="acme/coder-1", **extra):
    doc = {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "test-agent", "version": "1.0", "model_name": model},
        "steps": [{"step_id": i + 1, **step} for i, step in enumerate(steps)],
    }
    doc.update(extra)
    return doc


def _trial(tmp_path, doc, files=None, name="trial"):
    trial = tmp_path / name
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "trajectory.json").write_text(json.dumps(doc))
    for relative, content in (files or {}).items():
        path = trial / "agent" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content if isinstance(content, str) else json.dumps(content))
    return trial


def user(text="do it", **kw):
    return {"source": "user", "message": text, **kw}


def system(text="you are an agent", **kw):
    return {"source": "system", "message": text, **kw}


def agent(text="ok", **kw):
    return {"source": "agent", "message": text, **kw}


def call(call_id, name="bash", **arguments):
    return {"tool_call_id": call_id, "function_name": name, "arguments": arguments}


def results(*items):
    return {"results": [dict(item) for item in items]}


def boundary(kind, *contents, refs=None):
    result = [{"content": c} for c in contents]
    if refs:
        result.append({"subagent_trajectory_ref": refs})
    return system(
        "context management",
        observation={"results": result},
        extra={"context_management": {"type": "compaction", "boundary": kind}},
    )


# ---- Harbor fixtures ----


def test_find_trials_in_a_job_dir_and_a_trial_dir():
    trials = find_trials(HARBOR)
    assert [t.name for t in trials] == [
        "claude-code-hello",
        "terminus2-invalid-json",
        "terminus2-linear-history",
        "terminus2-summarization",
        "terminus2-timeout",
    ]
    assert find_trials(HARBOR / "terminus2-timeout") == [HARBOR / "terminus2-timeout"]


def test_claude_code_tool_call_append_only():
    (record,), dropped = _convert(HARBOR / "claude-code-hello")
    assert dropped == []
    assert "steps" not in record  # append-only shorthand
    assert _roles(record) == ["user", "assistant", "tool", "assistant"]
    (call_,) = record["messages"][1]["tool_calls"]
    assert call_["id"] == "toolu_01Rmm2GPxpGq9Er31axjfYRv"
    assert call_["function"]["name"] == "Write"
    assert json.loads(call_["function"]["arguments"]) == {"file_path": "/app/hello.txt", "content": "Hello, world!\n"}
    assert record["messages"][2]["tool_call_id"] == call_["id"]
    assert record["metadata"] == {
        "source": "atif",
        "trial": str(HARBOR / "claude-code-hello"),
        "trajectory": "agent/trajectory.json",
        "agent": "claude-code",
        "agent_version": "2.1.222",
        "model": "claude-opus-5",
    }


def test_terminus2_text_observations_and_reasoning():
    (record,), _ = _convert(HARBOR / "terminus2-invalid-json")
    assert _roles(record)[:3] == ["user", "assistant", "user"]
    assert record["messages"][2]["content"].startswith("Previous response had parsing")
    assert all("reasoning_content" in m for m in record["messages"] if m["role"] == "assistant")


def test_terminus2_compaction_follows_the_spec_rule():
    records, dropped = _convert(HARBOR / "terminus2-summarization")
    assert dropped == []
    by = _by_trajectory(records)
    assert list(by) == [
        "agent/trajectory.json",
        "agent/trajectory.summarization-1-summary.json",
        "agent/trajectory.summarization-1-questions.json",
        "agent/trajectory.summarization-1-answers.json",
    ]
    main = by["agent/trajectory.json"]
    steps = _steps(main)
    # Three replies before the boundary in one chain, then the context restarts: the boundary step
    # has no content, so the first post-boundary context is only the handoff user message.
    assert [s[1] for s in steps] == [1, 3, 5, 8, 10, 12, 14]
    handoff = steps[3][0]
    assert len(handoff) == 1 and main["messages"][handoff[0]]["content"].startswith("Here are the answers")
    assert all(train for _, _, train in steps)
    assert "Performed context summarization" not in json.dumps(main["messages"])


def test_terminus2_summarization_subagents_train_only_their_new_call():
    records, _ = _convert(HARBOR / "terminus2-summarization")
    by = _by_trajectory(records)
    for name, completion in [("summary", 6), ("answers", 8)]:
        record = by[f"agent/trajectory.summarization-1-{name}.json"]
        assert record["metadata"]["parent"] == "agent/trajectory.json"
        assert record["metadata"]["agent"] == f"terminus-2-summarization-{name}"
        # copied context is in the prompt, never a completion
        assert _steps(record) == [(list(range(completion)), completion, True)]
    assert "steps" not in by["agent/trajectory.summarization-1-questions.json"]


def test_terminus2_linear_history_continuation_is_one_record():
    records, dropped = _convert(HARBOR / "terminus2-linear-history")
    (record,) = records
    steps = _steps(record)
    first_after = steps[3]
    copied = [record["messages"][i] for i in first_after[0]]
    assert [m["role"] for m in copied] == ["user", "user", "assistant", "user"]
    assert copied[2]["content"].startswith("I have the following questions")
    assert copied[2] not in [record["messages"][c] for _, c, _ in steps]  # copied reply never trained
    assert len(steps) == 3 + 4
    # Harbor's golden data leaves out the summarization files this trajectory names.
    assert sorted((d.reason, d.detail) for d in dropped) == [
        ("missing_subagent", "trajectory.summarization-1-answers.json"),
        ("missing_subagent", "trajectory.summarization-1-questions.json"),
        ("missing_subagent", "trajectory.summarization-1-summary.json"),
    ]


def test_token_ids_are_copied_only_on_request():
    (record,), _ = _convert(HARBOR / "terminus2-timeout")
    assert not any("completion_token_ids" in m for m in record["messages"])
    (record,), _ = _convert(HARBOR / "terminus2-timeout", token_ids=True)
    ids = [m["completion_token_ids"] for m in record["messages"] if m["role"] == "assistant"]
    assert len(ids) == 3 and all(ids)


def test_step_model_names_match_without_provider_prefix():
    # agent.model_name is "openai/gpt-4o" and each step says "gpt-4o": all trained
    (record,), _ = _convert(HARBOR / "terminus2-timeout")
    assert all(train for _, _, train in _steps(record))
    _, dropped = _convert(HARBOR / "terminus2-timeout", model="some-other-model")
    assert [d.reason for d in dropped] == ["no_trained_step"]


# ---- the mapping, on written trial dirs ----


def test_system_prompt_tools_and_render_settings(tmp_path):
    doc = _doc([system(), user(), agent("done")])
    doc["agent"]["tool_definitions"] = TOOLS
    (record,), _ = _convert(_trial(tmp_path, doc), chat_template_kwargs={"enable_thinking": True})
    assert _roles(record) == ["system", "user", "assistant"]
    assert record["tools"] == TOOLS
    assert record["chat_template_kwargs"] == {"enable_thinking": True}


def test_parallel_tool_calls_keep_result_order(tmp_path):
    step = agent(
        "two calls",
        tool_calls=[call("a", cmd="ls"), call("b", cmd="pwd")],
        observation=results({"source_call_id": "b", "content": "/"}, {"source_call_id": "a", "content": "x"}),
    )
    (record,), _ = _convert(_trial(tmp_path, _doc([user(), step, agent("done")])))
    assert [(m["role"], m.get("tool_call_id")) for m in record["messages"][2:4]] == [("tool", "b"), ("tool", "a")]
    assert "steps" not in record


def test_tool_call_without_result_is_kept(tmp_path):
    step = agent("call", tool_calls=[call("a")], observation=results({"content": "terminal says hi"}))
    (record,), _ = _convert(_trial(tmp_path, _doc([user(), step, agent("done")])))
    assert _roles(record) == ["user", "assistant", "user", "assistant"]


def test_unknown_source_call_id_is_refused(tmp_path):
    step = agent("call", tool_calls=[call("a")], observation=results({"source_call_id": "zz", "content": "?"}))
    records, dropped = _convert(_trial(tmp_path, _doc([user(), step])))
    assert records == [] and [d.reason for d in dropped] == ["unknown_tool_call"]


def test_text_content_parts_join_and_media_is_refused(tmp_path):
    parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    (record,), _ = _convert(_trial(tmp_path, _doc([user(parts), agent()])))
    assert record["messages"][0]["content"] == "ab"
    image = [{"type": "image", "source": {"media_type": "image/png", "path": "images/x.png"}}]
    _, dropped = _convert(_trial(tmp_path, _doc([user(image), agent()]), name="t2"))
    assert [d.reason for d in dropped] == ["multimodal"]


def test_llm_call_count(tmp_path):
    dispatch = agent("", llm_call_count=0, tool_calls=[call("a")], observation=results({"source_call_id": "a"}))
    (record,), _ = _convert(_trial(tmp_path, _doc([user(), dispatch, agent("after")])))
    assert _steps(record) == [([0, 1, 2], 3, True)]  # the dispatch is context only
    _, dropped = _convert(_trial(tmp_path, _doc([user(), agent(llm_call_count=2)]), name="t2"))
    assert [d.reason for d in dropped] == ["aggregated_step"]


def test_other_model_steps_are_context_only(tmp_path):
    doc = _doc([user(), agent("strong", model_name="big/oracle-9"), user("next"), agent("ours")])
    (record,), _ = _convert(_trial(tmp_path, doc))
    assert _steps(record) == [([0], 1, False), ([0, 1, 2], 3, True)]


def test_copied_context_is_never_trained(tmp_path):
    doc = _doc([user(is_copied_context=True), agent("old", is_copied_context=True), user("go on"), agent("new")])
    (record,), _ = _convert(_trial(tmp_path, doc))
    assert _steps(record) == [([0, 1, 2], 3, True)]


def test_only_copied_steps_give_no_record(tmp_path):
    _, dropped = _convert(_trial(tmp_path, _doc([user(is_copied_context=True), agent(is_copied_context=True)])))
    assert [d.reason for d in dropped] == ["no_trained_step"]


# ---- context management ----


def test_replace_boundary_with_summary(tmp_path):
    doc = _doc([system(), user(), agent("a1"), boundary("replace", "SUMMARY"), user("u2"), agent("a2")])
    (record,), _ = _convert(_trial(tmp_path, doc))
    messages = record["messages"]
    steps = _steps(record)
    assert steps[0] == ([0, 1], 2, True)
    context = [messages[i] for i in steps[1][0]]
    assert [(m["role"], m["content"]) for m in context] == [("user", "SUMMARY"), ("user", "u2")]


def test_append_boundary_adds_to_the_context(tmp_path):
    doc = _doc([user(), agent("a1"), boundary("append", "NOTE"), agent("a2")])
    (record,), _ = _convert(_trial(tmp_path, doc))
    assert "steps" not in record  # still append-only
    assert record["messages"][2] == {"role": "user", "content": "NOTE"}


@pytest.mark.parametrize("kind", ["truncate", None, "sliding"])
def test_undetermined_boundary_is_refused(tmp_path, kind):
    step = boundary("replace")
    step["extra"]["context_management"] = {"type": "pruning"} if kind is None else {"boundary": kind}
    _, dropped = _convert(_trial(tmp_path, _doc([user(), agent(), step, user(), agent()])))
    assert [d.reason for d in dropped] == ["context_undetermined"]


def test_replace_then_agent_with_no_context_is_refused(tmp_path):
    _, dropped = _convert(_trial(tmp_path, _doc([user(), agent(), boundary("replace"), agent()])))
    assert [d.reason for d in dropped] == ["empty_context"]


def test_missing_continuation_refuses_the_record(tmp_path):
    doc = _doc([user(), agent()], continued_trajectory_ref="trajectory.cont-1.json")
    _, dropped = _convert(_trial(tmp_path, doc))
    assert [(d.reason, d.detail) for d in dropped] == [("missing_continuation", "trajectory.cont-1.json")]


def test_subagents_survive_a_missing_continuation(tmp_path):
    doc = _doc([user(), _delegating([{"trajectory_path": "sub.json"}])], continued_trajectory_ref="cont.json")
    records, dropped = _convert(_trial(tmp_path, doc, {"sub.json": _doc([user("s"), agent("s done")])}))
    assert [r["metadata"]["trajectory"] for r in records] == ["agent/sub.json"]
    assert [d.reason for d in dropped] == ["missing_continuation"]


def test_continuation_file_is_not_a_main_trajectory(tmp_path):
    cont = _doc([user("carried", is_copied_context=True), agent("a2")])
    doc = _doc([user(), agent("a1")], continued_trajectory_ref="trajectory.cont-1.json")
    records, _ = _convert(_trial(tmp_path, doc, {"trajectory.cont-1.json": cont}))
    (record,) = records
    assert [m["content"] for m in record["messages"]] == ["do it", "a1", "carried", "a2"]


# ---- subagents ----


def _delegating(refs):
    return agent(
        "delegate",
        tool_calls=[call("t1", name="task")],
        observation={"results": [{"source_call_id": "t1", "content": "sub result", "subagent_trajectory_ref": refs}]},
    )


def test_embedded_and_file_subagents(tmp_path):
    embedded = _doc([user("sub task"), agent("sub done")], trajectory_id="sub-1")
    embedded["agent"]["tool_definitions"] = TOOLS
    file_sub = _doc([user("file task"), agent("file done")])
    doc = _doc(
        [user(), _delegating([{"trajectory_id": "sub-1"}, {"trajectory_path": "subagents/x.json"}]), agent("end")],
        subagent_trajectories=[embedded],
    )
    records, dropped = _convert(_trial(tmp_path, doc, {"subagents/x.json": file_sub}))
    assert dropped == []
    by = _by_trajectory(records)
    assert list(by) == ["agent/trajectory.json", "agent/trajectory.json#sub-1", "agent/subagents/x.json"]
    assert by["agent/trajectory.json#sub-1"]["tools"] == TOOLS
    assert by["agent/subagents/x.json"]["metadata"]["parent"] == "agent/trajectory.json"
    assert "tools" not in by["agent/trajectory.json"]
    assert by["agent/trajectory.json"]["messages"][2] == {
        "role": "tool",
        "tool_call_id": "t1",
        "content": "sub result",
    }


def test_nested_subagent(tmp_path):
    inner = _doc([user("inner"), agent("inner done")], trajectory_id="inner")
    outer = _doc([user("outer"), _delegating([{"trajectory_id": "inner"}])], trajectory_id="outer")
    outer["subagent_trajectories"] = [inner]
    doc = _doc([user(), _delegating([{"trajectory_id": "outer"}])], subagent_trajectories=[outer])
    records, dropped = _convert(_trial(tmp_path, doc))
    assert dropped == []
    by = _by_trajectory(records)
    assert list(by) == ["agent/trajectory.json", "agent/trajectory.json#outer", "agent/trajectory.json#outer#inner"]
    assert by["agent/trajectory.json#outer#inner"]["metadata"]["parent"] == "agent/trajectory.json#outer"


def test_unresolvable_subagents_drop_only_themselves(tmp_path):
    refs = [{"trajectory_id": "nope"}, {"trajectory_path": "gone.json"}, {"session_id": "s"}]
    records, dropped = _convert(_trial(tmp_path, _doc([user(), _delegating(refs)])))
    assert len(records) == 1
    assert [d.reason for d in dropped] == ["missing_subagent", "missing_subagent", "unresolvable_subagent"]


def test_malformed_subagent_refs_are_reported_not_raised(tmp_path):
    refs = [{"trajectory_path": 1}, {"trajectory_id": ["x"]}, "not a ref"]
    doc = _doc([user(), _delegating(refs)], subagent_trajectories=7)
    records, dropped = _convert(_trial(tmp_path, doc))
    assert len(records) == 1
    assert [d.reason for d in dropped] == ["unresolvable_subagent"] * 3


def test_subagent_referenced_twice_is_converted_once(tmp_path):
    sub = _doc([user("s"), agent("s done")])
    ref = [{"trajectory_path": "sub.json"}]
    doc = _doc([user(), _delegating(ref), user("again"), _delegating(copy.deepcopy(ref))])
    doc["steps"][3]["tool_calls"][0]["tool_call_id"] = "t2"
    doc["steps"][3]["observation"]["results"][0]["source_call_id"] = "t2"
    records, _ = _convert(_trial(tmp_path, doc, {"sub.json": sub}))
    assert len(records) == 2


def test_subagents_of_a_refused_parent_are_still_converted(tmp_path):
    sub = _doc([user("s"), agent("s done")])
    doc = _doc([user(), _delegating([{"trajectory_path": "sub.json"}]), agent(llm_call_count=3)])
    records, dropped = _convert(_trial(tmp_path, doc, {"sub.json": sub}))
    assert [r["metadata"]["trajectory"] for r in records] == ["agent/sub.json"]
    assert [d.reason for d in dropped] == ["aggregated_step"]


# ---- trial layout and input checks ----


def test_multi_stage_trial(tmp_path):
    trial = tmp_path / "trial"
    for stage in ["a", "b"]:
        (trial / "steps" / stage / "agent").mkdir(parents=True)
        (trial / "steps" / stage / "agent" / "trajectory.json").write_text(json.dumps(_doc([user(stage), agent()])))
    assert find_trials(tmp_path) == [trial]
    records, _ = _convert(trial)
    assert [r["metadata"]["trajectory"] for r in records] == [
        "steps/a/agent/trajectory.json",
        "steps/b/agent/trajectory.json",
    ]


def test_bad_files_are_reported(tmp_path):
    _, dropped = _convert(_trial(tmp_path, {"schema_version": "OTHER-v2", "agent": {}, "steps": []}))
    assert [d.reason for d in dropped] == ["schema_version"]
    trial = _trial(tmp_path, _doc([user(), agent()]), name="t2")
    (trial / "agent" / "trajectory.json").write_text("{not json")
    _, dropped = _convert(trial)
    assert [d.reason for d in dropped] == ["unreadable"]


def test_malformed_steps_are_reported_not_raised(tmp_path):
    doc = _doc([user(), agent()])
    doc["steps"].append("not a step")
    doc["steps"].append({"source": "agent", "message": "x", "observation": {"results": ["bad", {"content": 1}]}})
    records, dropped = _convert(_trial(tmp_path, doc))
    assert records == [] and [d.reason for d in dropped] == ["invalid_trajectory"]


# ---- writing records ----


def test_write_records_validates_and_reports(tmp_path):
    good = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    bad = {"messages": [{"role": "user", "content": "q"}], "metadata": {"trajectory": "x"}}
    out = tmp_path / "records.jsonl"
    report = write_records([good, bad, Dropped("y", "multimodal", "image")], out)
    assert report.records == 1
    assert [json.loads(line) for line in out.read_text().splitlines()] == [good]
    written = json.loads(report_path(out).read_text())
    assert written["dropped"] == {"invalid_record": 1, "multimodal": 1}
    assert written["examples"]["multimodal"] == [{"source": "y", "detail": "image"}]


def test_cli_on_the_harbor_job_dir(tmp_path, capsys):
    out = tmp_path / "records.jsonl"
    atif.main(["--input", str(HARBOR), "--input", str(tmp_path / "empty"), "--output", str(out)])
    lines = out.read_text().splitlines()
    assert len(lines) == 8  # 1 + 1 + 1 + (1 + 3 subagents) + 1
    report = json.loads(report_path(out).read_text())
    assert report == {
        "records": 8,
        "dropped": {"missing_subagent": 3, "no_trial": 1},
        "examples": report["examples"],
    }
    assert "8 records" in capsys.readouterr().out


def test_harbor_records_prepare_with_the_offline_tokenizer(tmp_path):
    pytest.importorskip("sglang")
    from miles_ext.sft.prepare import prepare
    from miles_ext.sft.render import SftRenderer
    from miles_ext.tests.test_sft_render import _qwen_tito

    out = tmp_path / "records.jsonl"
    atif.main(["--input", str(HARBOR), "--output", str(out)])
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    counts = prepare(
        inputs=[str(out)], output=prepared, renderer=SftRenderer(_qwen_tito()), processors=[], max_tokens=None
    )
    assert (counts.records, counts.records_skipped) == (8, 0), counts.errors
    assert counts.trained_tokens > 0
