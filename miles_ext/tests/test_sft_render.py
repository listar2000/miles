"""TITO-aware SFT rendering (ext-048, hook-017), on offline tokenizers (``sft_fixtures``):

    python -m pytest miles_ext/tests/test_sft_render.py
"""

import json

import pytest
from miles_ext.sft.record import parse_record
from miles_ext.sft.render import SftRenderer
from miles_ext.tests.sft_fixtures import deepseek_v41_tokenizer, qwen_tokenizer

from miles.rollout.session.config import SessionServerConfig
from miles.rollout.session.types import SessionRecord
from miles.rollout.session.v2.session_state import (
    SessionStateV2,
    commit_generation,
    prepare_token_ids_and_request_args,
)
from miles.utils.chat_template_utils.tito_tokenizer import get_tito_tokenizer, resolve_fixed_chat_template

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
        },
    }
]
SYSTEM = {"role": "system", "content": "S1"}


def _call(i, cmd):
    return {"id": f"call_{i}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"cmd": cmd})}}


def _user(i):
    return {"role": "user", "content": f"U{i}"}


def _assistant(i, **extra):
    return {"role": "assistant", "content": f"A{i}", "reasoning_content": f"R{i}", **extra}


# One user query, then a tool loop: what a SWE or Harbor agent trajectory looks like.
AGENT = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "Fix the bug."},
    {
        "role": "assistant",
        "content": "Let me look.",
        "reasoning_content": "List files.",
        "tool_calls": [_call(0, "ls")],
    },
    {"role": "tool", "tool_call_id": "call_0", "content": "a.py"},
    {"role": "assistant", "content": "", "reasoning_content": "Read a.py.", "tool_calls": [_call(1, "cat a.py")]},
    {"role": "tool", "tool_call_id": "call_1", "content": "print(1)"},
    {"role": "assistant", "content": "Fixed.", "reasoning_content": "Done."},
]
# Three user queries: (S1, U1, A1, U2, A2, U3, A3).
CHAT = [SYSTEM, _user(1), _assistant(1), _user(2), _assistant(2), _user(3), _assistant(3)]


def _qwen_tito():
    template_path, fixed_kwargs = resolve_fixed_chat_template("qwennext")
    assert template_path.endswith("qwen3_thinking_2507_and_next_fixed.jinja")
    return get_tito_tokenizer(qwen_tokenizer(), "qwennext", chat_template_kwargs=fixed_kwargs)


def _deepseek_tito(tmp_path):
    _, fixed_kwargs = resolve_fixed_chat_template("deepseekv41")
    return get_tito_tokenizer(deepseek_v41_tokenizer(tmp_path), "deepseekv41", chat_template_kwargs=fixed_kwargs)


@pytest.fixture
def qwen():
    return _qwen_tito()


def _trained_text(tokenizer, sample) -> list[str]:
    spans, current = [], []
    for token, mask in zip(sample.token_ids, sample.loss_mask, strict=True):
        if mask:
            current.append(token)
        elif current:
            spans.append(tokenizer.decode(current))
            current = []
    if current:
        spans.append(tokenizer.decode(current))
    return spans


def _full_render(tito, renderer, record) -> str:
    args = renderer._template_args(record)
    return tito.apply_chat_template(list(record.messages), add_generation_prompt=False, template_args=args)


# --- hook-017: tokenize_completion -----------------------------------------------------------


def test_qwen_completion_is_the_body_after_the_generation_prompt(qwen):
    ids = qwen.tokenize_completion(_assistant(1), template_args=qwen.default_template_args())
    assert qwen.tokenizer.decode(ids) == "R1\n</think>\n\nA1<|im_end|>"
    assert ids[-1] == qwen.tokenizer.convert_tokens_to_ids("<|im_end|>"), "the template's trailing \\n is not sampled"


def test_qwen_completion_renders_tool_calls(qwen):
    message = {"role": "assistant", "content": "", "reasoning_content": "r", "tool_calls": [_call(0, "ls")]}
    text = qwen.tokenizer.decode(qwen.tokenize_completion(message, template_args=qwen.default_template_args(TOOLS)))
    assert text == 'r\n</think>\n\n<tool_call>\n{"name": "bash", "arguments": {"cmd": "ls"}}\n</tool_call><|im_end|>'


def test_deepseek_completion_ends_at_end_of_sentence(tmp_path):
    tito = _deepseek_tito(tmp_path)
    args = SftRenderer(tito)._template_args(parse_record({"messages": CHAT}))
    assert tito.tokenizer.decode(tito.tokenize_completion(_assistant(1), template_args=args)) == (
        "R1</think>A1<｜end▁of▁sentence｜>"
    )


# --- session-server layout (thinking_retention="all") --------------------------------------


def test_append_only_agent_trajectory_is_one_sample_equal_to_a_full_render(qwen):
    record = parse_record({"messages": AGENT, "tools": TOOLS})
    renderer = SftRenderer(qwen)
    (sample,) = renderer.render(record)
    assert sample.trained_steps == (0, 1, 2)
    # The session server keeps every sampled turn verbatim; the fixed template renders the same text,
    # plus the "\n" after the last <|im_end|>, which the engine never samples.
    assert qwen.tokenizer.decode(sample.token_ids) + "\n" == _full_render(qwen, renderer, record)
    assert _trained_text(qwen.tokenizer, sample) == [
        'List files.\n</think>\n\nLet me look.\n<tool_call>\n{"name": "bash", "arguments": {"cmd": "ls"}}\n'
        "</tool_call><|im_end|>",
        'Read a.py.\n</think>\n\n<tool_call>\n{"name": "bash", "arguments": {"cmd": "cat a.py"}}\n'
        "</tool_call><|im_end|>",
        "Done.\n</think>\n\nFixed.<|im_end|>",
    ]


def test_tool_schema_is_rendered_into_the_prompt(qwen):
    with_tools, without = (
        SftRenderer(qwen).render(parse_record({"messages": AGENT, **extra}))[0] for extra in ({"tools": TOOLS}, {})
    )
    assert "Run a shell command." in qwen.tokenizer.decode(with_tools.token_ids)
    assert "Run a shell command." not in qwen.tokenizer.decode(without.token_ids)


def test_harness_that_keeps_one_past_exchange_splits_where_history_is_rewritten(qwen):
    """C1 = (S1, U1), C2 = (S1, U1, A1, U2), C3 = (S1, U2, A2, U3): C2 extends C1 + A1, C3 does not."""
    steps = [
        {"context": [0, 1], "completion": 2},
        {"context": [0, 1, 2, 3], "completion": 4},
        {"context": [0, 3, 4, 5], "completion": 6},
    ]
    first, second = SftRenderer(qwen).render(parse_record({"messages": CHAT, "steps": steps}))
    assert first.trained_steps == (0, 1) and second.trained_steps == (2,)
    assert [t.split("</think>")[1].strip() for t in _trained_text(qwen.tokenizer, first)] == [
        "A1<|im_end|>",
        "A2<|im_end|>",
    ]
    # C3 is rendered whole: A2 sits in its prompt untrained, only A3 is trained.
    expected_prompt = qwen.apply_chat_template([SYSTEM, _user(2), _assistant(2), _user(3)], add_generation_prompt=True)
    text = qwen.tokenizer.decode(second.token_ids)
    assert text.startswith(expected_prompt) and "U1" not in text
    assert _trained_text(qwen.tokenizer, second) == ["R3\n</think>\n\nA3<|im_end|>"]


def test_a_reply_shared_by_two_branches_is_trained_once(qwen):
    """Two retries of the second call (A2 and an alternative) both continue the first step."""
    messages = [*CHAT[:5], _assistant(9)]
    steps = [
        {"context": [0, 1], "completion": 2},
        {"context": [0, 1, 2, 3], "completion": 4},
        {"context": [0, 1, 2, 3], "completion": 5},
    ]
    samples = SftRenderer(qwen).render(parse_record({"messages": messages, "steps": steps}))
    assert [s.trained_steps for s in samples] == [(0, 1), (2,)]
    assert "R1" in qwen.tokenizer.decode(samples[1].token_ids), "the shared first reply is still context"


def test_untrained_steps_are_context_only(qwen):
    steps = [{"context": [0, 1], "completion": 2, "train": False}, {"context": [0, 1, 2, 3], "completion": 4}]
    (sample,) = SftRenderer(qwen).render(parse_record({"messages": CHAT[:5], "steps": steps}))
    assert sample.trained_steps == (1,)
    assert _trained_text(qwen.tokenizer, sample) == ["R2\n</think>\n\nA2<|im_end|>"]


def test_raw_completion_tokens_are_used_verbatim(qwen):
    """Sampled tokens beat a re-render, e.g. the blank line before <tool_call> a parsed message loses."""
    sampled = qwen.tokenizer.encode("R1\n</think>\n\nA1 ", add_special_tokens=False) + [
        qwen.tokenizer.convert_tokens_to_ids("<|im_end|>")
    ]
    messages = [SYSTEM, _user(1), _assistant(1, completion_token_ids=sampled), _user(2), _assistant(2)]
    (sample,) = SftRenderer(qwen).render(parse_record({"messages": messages}))
    assert _trained_text(qwen.tokenizer, sample)[0] == "R1\n</think>\n\nA1 <|im_end|>"
    assert "U2" in qwen.tokenizer.decode(sample.token_ids), "the next turn is merged after the raw tokens"


def test_deepseek_append_only_trajectory_matches_the_official_encoder(tmp_path):
    tito = _deepseek_tito(tmp_path)
    record = parse_record({"messages": CHAT})
    renderer = SftRenderer(tito)
    (sample,) = renderer.render(record)
    assert sample.trained_steps == (0, 1, 2)
    assert tito.tokenizer.decode(sample.token_ids) == _full_render(tito, renderer, record)
    assert _trained_text(tito.tokenizer, sample) == [f"R{i}</think>A{i}<｜end▁of▁sentence｜>" for i in (1, 2, 3)]


# --- plain-server layout (thinking_retention="template") -----------------------------------


def test_template_mode_splits_where_the_native_template_drops_reasoning(qwen):
    """With old reasoning dropped at each new user query, every turn's prompt is a new render."""
    record = parse_record({"messages": CHAT, "chat_template_kwargs": {"clear_thinking": True}})
    samples = SftRenderer(qwen, thinking_retention="template", native_tokenizer=qwen_tokenizer()).render(record)
    assert [s.trained_steps for s in samples] == [(0,), (1,), (2,)]
    last = qwen.tokenizer.decode(samples[2].token_ids)
    assert "R1" not in last and "R2" not in last and "<|im_start|>assistant\nA1<|im_end|>" in last


def test_template_mode_on_a_tool_loop_matches_the_session_layout(qwen):
    """A single user query drops nothing, so a plain server and the session server see the same tokens."""
    record = parse_record({"messages": AGENT, "tools": TOOLS, "chat_template_kwargs": {"clear_thinking": True}})
    (session,) = SftRenderer(qwen).render(record)
    (plain,) = SftRenderer(qwen, thinking_retention="template", native_tokenizer=qwen_tokenizer()).render(record)
    assert plain.token_ids == session.token_ids and plain.loss_mask == session.loss_mask


def test_template_mode_needs_the_native_tokenizer(qwen):
    with pytest.raises(ValueError, match="native_tokenizer"):
        SftRenderer(qwen, thinking_retention="template")
    with pytest.raises(ValueError, match="thinking_retention"):
        SftRenderer(qwen, thinking_retention="tool_cycle")


# --- parity with the session server's own serving functions --------------------------------

_SERVER_CONFIG = dict(
    host="127.0.0.1",
    port=0,
    instance_id=None,
    backend_url="http://127.0.0.1:0",
    timeout=None,
    hf_checkpoint=None,
    chat_template_path=None,
    apply_chat_template_kwargs=None,
    use_rollout_routing_replay=False,
    use_rollout_indexer_replay=False,
    use_sampling_support_replay=False,
    sglang_speculative_algorithm=None,
    num_layers=None,
    moe_router_topk=None,
    save_debug_trajectory_data=None,
    lora_rank=0,
    lora_adapter_path=None,
    lora_train_only=False,
    use_session_server="v2",
    session_message_matcher="strict",
    pause_generation_mode=None,
    session_sample_picker_path=None,
    session_sample_postprocessor_path=None,
    rollout_top_logprobs_num=0,
    rollout_sampling_logprobs_mode="raw",
    rollout_temperature=1.0,
)


def _serve(tito, tito_model, record):
    """Replay *record*'s steps as requests through session server v2, with an engine that samples
    exactly ``tokenize_completion``; return every leaf's token ids in commit order."""
    config = SessionServerConfig(tito_model=tito_model, **_SERVER_CONFIG)
    state = SessionStateV2()
    for k, step in enumerate(record.steps):
        context, reply = record.context_messages(step), record.messages[step.completion]
        body = {"messages": context, "chat_template_kwargs": dict(record.chat_template_kwargs)}
        if record.tools:
            body["tools"] = record.tools
        prepared, parent = prepare_token_ids_and_request_args(state, body, config=config, tito_tokenizer=tito)
        commit_generation(
            state,
            parent=parent,
            request_messages=context,
            assistant_message=reply,
            prompt_token_ids=prepared.body["input_ids"],
            completion_token_ids=tito.tokenize_completion(reply, template_args=prepared.template_args),
            max_trim_tokens=tito.max_trim_tokens,
            record=SessionRecord(
                timestamp=0.0, method="POST", path="/v1/chat/completions", request={}, response={}, status_code=200
            ),
            response_id=f"r{k}",
            finish_reason="stop",
            turn_args=prepared.body,
        )
    return [leaf.token_ids for leaf in state.tree.leaves()]


_ONE_PAST_EXCHANGE = [
    {"context": [0, 1], "completion": 2},
    {"context": [0, 1, 2, 3], "completion": 4},
    {"context": [0, 3, 4, 5], "completion": 6},
]


@pytest.mark.parametrize(
    "raw",
    [
        {"messages": AGENT, "tools": TOOLS},
        {"messages": CHAT},
        {"messages": CHAT, "steps": _ONE_PAST_EXCHANGE},
        {
            "messages": [*CHAT[:5], _assistant(9)],
            "steps": [*_ONE_PAST_EXCHANGE[:2], {"context": [0, 1, 2, 3], "completion": 5}],
        },
    ],
    ids=["agent", "chat", "one-past-exchange", "retry-branch"],
)
@pytest.mark.parametrize("family", ["qwennext", "deepseekv41"])
def test_samples_equal_what_the_session_server_builds(tmp_path, family, raw):
    tito = _qwen_tito() if family == "qwennext" else _deepseek_tito(tmp_path)
    record = parse_record(raw)
    samples = SftRenderer(tito).render(record)
    assert [s.token_ids for s in samples] == _serve(tito, family, record)
