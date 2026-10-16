"""[fork fix-005] A DeepSeek V4.1 reply whose text holds the image placeholder has no
canonical rendering; the session-mismatch diagnostic skips it instead of logging a failure."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
from sglang.srt.entrypoints.openai import encoding_dsv41

from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.utils.chat_template_utils import apply_chat_template
from miles.utils.chat_template_utils.deepseek import UnrenderableMessagesError


class _FakeTokenizer:
    def __init__(self, name_or_path: str):
        self.name_or_path = name_or_path


def _messages(**assistant):
    return [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "", **assistant}]


@pytest.mark.parametrize("key", ["content", "reasoning_content"])
def test_the_v41_bridge_names_the_image_placeholder_refusal(tmp_path, key):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}), encoding="utf-8")
    text = f"the token {encoding_dsv41.IMAGE_PLACEHOLDER} shows up"

    with pytest.raises(UnrenderableMessagesError, match=key):
        apply_chat_template(_messages(**{key: text}), tokenizer=_FakeTokenizer(str(tmp_path)))


class _RefusingTITOTokenizer:
    def create_comparator(self):
        return None

    def apply_chat_template(self, messages, **kwargs):
        raise UnrenderableMessagesError("refused")


def test_the_session_mismatch_diagnostic_skips_an_unrenderable_session(caplog):
    registry = SessionRegistry(tokenizer=None, tito_tokenizer=_RefusingTITOTokenizer())
    session = SimpleNamespace(token_ids=[1, 2], messages=_messages(), turn_args={})

    with caplog.at_level(logging.INFO):
        assert registry.compute_session_mismatch(session) is None

    assert "tito_session_mismatch skipped: refused" in caplog.text
