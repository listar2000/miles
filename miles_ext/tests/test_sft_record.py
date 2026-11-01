"""Test SFT record validation and the built-in post-processors."""

import copy

import pytest

from miles_ext.sft import postprocess as P
from miles_ext.sft.record import RecordError, Step, parse_record

SYSTEM = {"role": "system", "content": "s"}
USER = {"role": "user", "content": "u"}


def _assistant(text="a", **extra):
    return {"role": "assistant", "content": text, **extra}


def test_default_steps_train_every_assistant_after_its_prefix():
    record = parse_record({"messages": [SYSTEM, USER, _assistant(), {"role": "user", "content": "u2"}, _assistant()]})
    assert record.steps == (Step(context=(0, 1), completion=2), Step(context=(0, 1, 2, 3), completion=4))
    assert record.chat_template_kwargs == {} and record.tools is None


def test_explicit_steps_point_into_the_message_store():
    summary = {"role": "user", "content": "summary of turns 1-2"}
    raw = {
        "messages": [SYSTEM, USER, _assistant(), summary, _assistant()],
        "steps": [{"context": [0, 1], "completion": 2, "train": False}, {"context": [0, 3], "completion": 4}],
    }
    record = parse_record(raw)
    assert record.steps[0].train is False
    assert record.context_messages(record.steps[1]) == [SYSTEM, summary]


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ({"messages": [SYSTEM, USER]}, "no assistant"),
        ({"messages": [_assistant()]}, "no assistant"),
        ({"messages": [SYSTEM, USER, _assistant()], "extra": 1}, "unknown record keys"),
        ({"messages": [{"role": "developer", "content": "x"}, _assistant()]}, "role"),
        ({"messages": [SYSTEM, {"role": "user", "content": None}, _assistant()]}, "content"),
        ({"messages": [SYSTEM, USER, _assistant()], "steps": [{"context": [0], "completion": 1}]}, "not an assistant"),
        ({"messages": [SYSTEM, USER, _assistant()], "steps": [{"context": [0, 9], "completion": 2}]}, "outside"),
        ({"messages": [SYSTEM, USER, _assistant()], "steps": [{"context": [2], "completion": 2}]}, "also in its"),
        (
            {"messages": [SYSTEM, USER, _assistant()], "steps": [{"context": [{"role": "user"}], "completion": 2}]},
            "indices",
        ),
        ({"messages": [SYSTEM, USER, _assistant(completion_token_ids=[])]}, "completion_token_ids"),
    ],
)
def test_invalid_records_are_rejected(raw, error):
    with pytest.raises(RecordError, match=error):
        parse_record(raw)


def test_tool_call_arguments_must_be_a_json_string():
    call = {"id": "c", "type": "function", "function": {"name": "f", "arguments": {"x": 1}}}
    with pytest.raises(RecordError, match="JSON string"):
        parse_record({"messages": [SYSTEM, USER, _assistant(tool_calls=[call])]})
    call["function"]["arguments"] = "{not json"
    with pytest.raises(RecordError, match="not JSON"):
        parse_record({"messages": [SYSTEM, USER, _assistant(tool_calls=[call])]})


def test_normalize_openai_messages():
    raw = {
        "messages": [
            SYSTEM,
            USER,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": '[{"function": {"name": "f", "arguments": {"x": 1}}}]',
            },
            {"role": "assistant", "content": "done", "tool_calls": []},
        ]
    }
    before = copy.deepcopy(raw)
    out = P.normalize_openai_messages(raw)
    assert raw == before, "a processor must not modify its input"
    assert out["messages"][2]["content"] == ""
    assert out["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"x": 1}'
    assert "tool_calls" not in out["messages"][3]
    parse_record(out)


def test_split_inline_think():
    raw = {"messages": [SYSTEM, USER, _assistant("<think>\nplan\n</think>\n\nanswer"), _assistant("plain")]}
    out = P.split_inline_think(raw)
    assert out["messages"][2] == {"role": "assistant", "content": "answer", "reasoning_content": "plan"}
    assert out["messages"][3] == _assistant("plain")
    kept = {"messages": [SYSTEM, USER, _assistant("x</think>y", reasoning_content="r")]}
    assert P.split_inline_think(kept) == kept


def test_strip_trailing_whitespace_drops_stale_raw_tokens():
    raw = {
        "messages": [
            SYSTEM,
            USER,
            _assistant("a  \n", completion_token_ids=[1]),
            _assistant("b", completion_token_ids=[2]),
        ]
    }
    out = P.strip_assistant_trailing_whitespace(raw)
    assert out["messages"][2] == _assistant("a")
    assert out["messages"][3] == _assistant("b", completion_token_ids=[2])


def test_processors_load_by_dotted_path_and_run_in_order():
    processors = P.load_processors(
        [
            "miles_ext.sft.postprocess.split_inline_think",
            "miles_ext.sft.postprocess.strip_assistant_trailing_whitespace",
        ]
    )
    out = P.apply_processors({"messages": [SYSTEM, USER, _assistant("<think>r</think>a \n")]}, processors)
    assert out["messages"][2] == {"role": "assistant", "content": "a", "reasoning_content": "r"}
