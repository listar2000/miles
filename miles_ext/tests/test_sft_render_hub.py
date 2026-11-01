"""SFT rendering with real Hugging Face tokenizers (ext-048, hook-017). Needs the Hub (or its cache):

    python -m pytest miles_ext/tests/test_sft_render_hub.py

Offline tokenizers keep one token per byte; these tests add real BPE merges, where encoding a
reply on its own could differ from encoding it inside the conversation.
"""

import pytest
from huggingface_hub import snapshot_download

from miles_ext.sft.record import parse_record
from miles_ext.sft.render import load_renderer
from miles_ext.tests.test_sft_render import _ONE_PAST_EXCHANGE, AGENT, CHAT, TOOLS, _serve

# (repo, tito model): the checkpoints the fork's presets train (miles_ext/recipes/presets.py).
FAMILIES = {
    "qwennext": "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "deepseekv41": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
}
# DeepSeek families are detected from config.json in a local directory, so fetch the tokenizer files.
_TOKENIZER_FILES = ["config.json", "tokenizer*", "special_tokens_map.json"]


@pytest.fixture(scope="module", params=list(FAMILIES))
def family(request):
    try:
        checkpoint = snapshot_download(FAMILIES[request.param], allow_patterns=_TOKENIZER_FILES)
    except Exception as exc:  # no network and no cache: nothing to test against
        pytest.skip(f"{FAMILIES[request.param]} tokenizer unavailable: {exc}")
    return request.param, load_renderer(checkpoint, request.param)


def test_agent_trajectory_tokens_equal_one_encode_of_the_full_render(family):
    tito_model, renderer = family
    record = parse_record({"messages": AGENT, "tools": TOOLS})
    (sample,) = renderer.render(record)
    args = renderer._template_args(record)
    full = renderer.tito.apply_chat_template(list(record.messages), add_generation_prompt=False, template_args=args)
    encoded = renderer.tito.tokenizer.encode(full, add_special_tokens=False)
    # Qwen's template adds a "\n" after the last <|im_end|> that the engine never samples.
    assert (encoded[:-1] if tito_model == "qwennext" else encoded) == sample.token_ids


@pytest.mark.parametrize("raw", [{"messages": AGENT, "tools": TOOLS}, {"messages": CHAT, "steps": _ONE_PAST_EXCHANGE}])
def test_samples_equal_what_the_session_server_builds(family, raw):
    tito_model, renderer = family
    record = parse_record(raw)
    assert [s.token_ids for s in renderer.render(record)] == _serve(renderer.tito, tito_model, record)
