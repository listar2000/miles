# [fork hook-017] TITOTokenizer.tokenize_completion on the real Qwen3-Thinking and DeepSeek V4.1 tokenizers
"""``tokenize_completion`` returns the tokens after the generation prompt through the stop token."""

from __future__ import annotations

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

import pytest
from huggingface_hub import snapshot_download

from miles.utils.chat_template_utils.tito_tokenizer import (
    extract_template_args,
    get_tito_tokenizer,
    resolve_fixed_chat_template,
)
from miles.utils.processing_utils import load_tokenizer

_CHECKPOINTS = {
    "qwennext": "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "deepseekv41": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
}
# DeepSeek families are detected from config.json in a local directory.
_TOKENIZER_FILES = ["config.json", "tokenizer*", "special_tokens_map.json"]
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
        },
    }
]
_TOOL_CALL = {
    "role": "assistant",
    "content": "",
    "reasoning_content": "r",
    "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}],
}
_REPLY = {"role": "assistant", "content": "A1", "reasoning_content": "R1"}


def _tito(tito_model: str):
    try:
        checkpoint = snapshot_download(_CHECKPOINTS[tito_model], allow_patterns=_TOKENIZER_FILES)
    except Exception as exc:  # no network and no cache
        pytest.skip(f"{_CHECKPOINTS[tito_model]} tokenizer unavailable: {exc}")
    template_path, fixed_kwargs = resolve_fixed_chat_template(tito_model)
    tokenizer = load_tokenizer(checkpoint, chat_template_path=template_path, trust_remote_code=True)
    return get_tito_tokenizer(tokenizer, tokenizer_type=tito_model, chat_template_kwargs=fixed_kwargs)


def _args(tito, tools=None):
    request = {"tools": tools, "chat_template_kwargs": {}}
    return extract_template_args(tito.resolve_request_args(request, turn_args=None))


@pytest.fixture(scope="module")
def qwen():
    return _tito("qwennext")


@pytest.fixture(scope="module")
def deepseek():
    return _tito("deepseekv41")


def test_qwen_completion_ends_at_im_end(qwen):
    ids = qwen.tokenize_completion(_REPLY, template_args=_args(qwen))
    assert qwen.tokenizer.decode(ids) == "R1\n</think>\n\nA1<|im_end|>"
    assert ids[-1] == qwen.tokenizer.convert_tokens_to_ids("<|im_end|>"), "the template's trailing \\n is not sampled"


def test_qwen_completion_renders_tool_calls(qwen):
    text = qwen.tokenizer.decode(qwen.tokenize_completion(_TOOL_CALL, template_args=_args(qwen, _TOOLS)))
    assert text == 'r\n</think>\n\n<tool_call>\n{"name": "bash", "arguments": {"cmd": "ls"}}\n</tool_call><|im_end|>'


def test_deepseek_completion_ends_at_end_of_sentence(deepseek):
    ids = deepseek.tokenize_completion(_REPLY, template_args=_args(deepseek))
    assert deepseek.tokenizer.decode(ids) == "R1</think>A1<｜end▁of▁sentence｜>"


@pytest.mark.parametrize("tito_model", list(_CHECKPOINTS))
def test_prompt_plus_completion_is_the_full_render(tito_model):
    tito = _tito(tito_model)
    args = _args(tito, _TOOLS)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    prompt = tito.apply_chat_template(messages, add_generation_prompt=True, template_args=args)
    full = tito.apply_chat_template(messages + [_TOOL_CALL], add_generation_prompt=False, template_args=args)
    completion = tito.tokenizer.decode(tito.tokenize_completion(_TOOL_CALL, template_args=args))
    assert full.startswith(prompt + completion)
    assert len(full) - len(prompt + completion) <= 1, "at most the one trailing template token is left out"
