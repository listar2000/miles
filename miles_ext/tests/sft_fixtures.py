"""Offline tokenizers for the SFT tests: no download, real rendering code.

- ``qwen_tokenizer``: a byte-level Hugging Face fast tokenizer (no merges) with Qwen3's
  special tokens and the repo's fixed Qwen3-Thinking-2507 template, so the Qwen3 TITO
  path runs exactly as for the real tokenizer, only with one token per byte.
- ``deepseek_v41_tokenizer``: one token per character over DeepSeek V4.1's official encoder
  (sglang ``encoding_dsv41``), as ``tests/fast/utils/chat_template_utils/test_deepseek_v4_1.py``
  does; special tokens get ids above the character range.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import bytes_to_unicode

from miles.utils.chat_template_utils.tito_tokenizer import TEMPLATE_DIR

QWEN_SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
QWEN_ADDED = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]
QWEN_TEMPLATE = TEMPLATE_DIR / "qwen3_thinking_2507_and_next_fixed.jinja"


def qwen_tokenizer(chat_template_path: Path | None = QWEN_TEMPLATE) -> PreTrainedTokenizerFast:
    vocab = {char: i for i, char in enumerate(bytes_to_unicode().values())}
    raw = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    raw.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tokenizer.add_special_tokens({"additional_special_tokens": QWEN_SPECIAL})
    tokenizer.add_tokens(QWEN_ADDED)
    if chat_template_path is not None:
        tokenizer.chat_template = chat_template_path.read_text()
    return tokenizer


class CharTokenizer:
    """One id per character; listed special tokens map to ids above the Unicode range."""

    def __init__(self, name_or_path: str, special_tokens: list[str]):
        self.name_or_path = name_or_path
        self._special = {token: 0x110000 + i for i, token in enumerate(special_tokens)}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i < 0x110000)

    def convert_tokens_to_ids(self, token):
        return self._special.get(token, -1)


def deepseek_v41_tokenizer(tmp_path: Path) -> CharTokenizer:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}), encoding="utf-8")
    return CharTokenizer(str(tmp_path), ["<｜User｜>", "<｜Assistant｜>", "<think>", "</think>"])
