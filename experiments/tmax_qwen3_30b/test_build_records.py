from build_records import convert_row
from miles_ext.sft.connectors.base import Dropped


def _row(reply):
    call = {"id": "c", "function": {"name": "bash", "arguments": {"command": "ls"}}}
    assistant = {"role": "assistant", "tool_calls": [call], **reply}
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, assistant]
    return {"messages": messages, "tools": "[]", "metadata": {"trial_name": "t"}}


def test_a_stray_think_close_is_removed_and_the_reply_trained():
    record = convert_row(_row({"content": "THOUGHT: x</think>", "reasoning_content": "r"}))
    assert record["messages"][2]["content"] == "THOUGHT: x\n\n"
    assert record["steps"][0]["train"] is True


def test_scaffolding_closing_the_reasoning_is_removed():
    record = convert_row(_row({"content": "", "reasoning_content": "r</parameter></function></tool_call>"}))
    assert record["messages"][2]["reasoning_content"] == "r\n"


def test_a_row_with_no_repairable_reply_is_dropped():
    row = _row({"content": "<|im_end|>", "reasoning_content": ""})
    assert isinstance(convert_row(row), Dropped)
