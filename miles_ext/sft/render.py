"""TITO-aware SFT rendering: a message list to the token ids an RL rollout would have produced.

An RL rollout through Miles' session server builds a trajectory's tokens turn by turn
(``miles/rollout/session/linear_trajectory.py``): the first prompt is the fixed template
rendered with the generation prompt, each assistant turn is the tokens the engine
sampled (ending at ``<|im_end|>``), and everything after it comes from
``TITOTokenizer.merge_tokens`` (the missing ``\\n`` plus the rendered observations and
the next generation prompt). ``render_trajectory`` replays that loop on a stored
message list, so SFT trains on the same token layout RL and serving see. The sampled
tokens are the one part it cannot know; it encodes the assistant turn's rendered text
on its own, which is what the engine produces for a canonically tokenized reply.

Loss mask: 1 on assistant tokens (reasoning, visible text, tool calls, ``<|im_end|>``),
0 on everything else, including each turn's ``<|im_start|>assistant\\n<think>\\n`` opener,
which belongs to the prompt in RL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from miles.utils.chat_template_utils.tito_tokenizer import (
    TITOTokenizer,
    get_tito_tokenizer,
    resolve_fixed_chat_template,
)

_PROBE_PREFIX = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


@dataclass
class Rendered:
    token_ids: list[int]
    loss_mask: list[int]
    # parity against one whole-conversation render of the same fixed template
    full_render_equal: bool = False
    full_render_text_equal: bool = False
    turns: int = 0
    notes: list[str] = field(default_factory=list)


def load_tito(hf_dir: str, tito_model: str = "qwennext") -> TITOTokenizer:
    """The tokenizer the session server uses: the model's tokenizer with the family's fixed template."""
    from miles.utils.processing_utils import load_tokenizer

    template_path, _ = resolve_fixed_chat_template(tito_model)
    tokenizer = load_tokenizer(hf_dir, chat_template_path=template_path, trust_remote_code=True)
    return get_tito_tokenizer(tokenizer, tito_model)


class TrajectoryRenderer:
    def __init__(self, tito: TITOTokenizer, tools: list[dict] | None = None):
        self.tito = tito
        self.tok = tito.tokenizer
        self.args = tito.default_template_args(tools)
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        gen = self._text(_PROBE_PREFIX, True)
        base = self._text(_PROBE_PREFIX, False)
        assert gen.startswith(base)
        self.gen_prompt = gen[len(base):]  # "<|im_start|>assistant\n<think>\n"

    def _text(self, messages: list[dict], add_generation_prompt: bool) -> str:
        return self.tito.apply_chat_template(messages, add_generation_prompt=add_generation_prompt,
                                             template_args=self.args)

    def completion_ids(self, assistant: dict) -> list[int]:
        """The tokens the engine samples for this assistant turn: after the generation prompt, up to and
        including ``<|im_end|>`` (the template's trailing ``\\n`` is never sampled)."""
        prompt = self._text(_PROBE_PREFIX, True)
        full = self._text(_PROBE_PREFIX + [assistant], False)
        if not full.startswith(prompt):
            raise ValueError("assistant render does not extend the generation prompt")
        body = full[len(prompt):]
        if not body.endswith("<|im_end|>\n"):
            raise ValueError(f"assistant render ends with {body[-20:]!r}")
        return self.tok.encode(body[: -len("<|im_end|>\n")], add_special_tokens=False) + [self.im_end]

    def render(self, messages: list[dict]) -> Rendered:
        first = next((i for i, m in enumerate(messages) if m["role"] == "assistant"), None)
        if first is None or first == 0:
            raise ValueError("needs a prompt before the first assistant message")
        ids = list(self.tito.apply_chat_template(messages[:first], add_generation_prompt=True, tokenize=True,
                                                 template_args=self.args))
        mask = [0] * len(ids)
        i, turns = first, 0
        while i < len(messages):
            assert messages[i]["role"] == "assistant"
            comp = self.completion_ids(messages[i])
            ids += comp
            mask += [1] * len(comp)
            turns += 1
            j = i + 1
            while j < len(messages) and messages[j]["role"] != "assistant":
                j += 1
            if j == len(messages):
                if j != i + 1:
                    raise ValueError("trajectory does not end with an assistant message")
                break
            # what the session server appends before the next turn: "\n", the observations, the generation prompt
            merged = self.tito.merge_tokens(messages[: i + 1], messages[:j], ids, template_args=self.args)
            assert merged[: len(ids)] == ids
            mask += [0] * (len(merged) - len(ids))
            ids = merged
            i = j
        out = Rendered(token_ids=ids, loss_mask=mask, turns=turns)
        full_text = self._text(messages, False)
        out.full_render_text_equal = full_text == self.tok.decode(ids) + "\n"
        out.full_render_equal = self.tok.encode(full_text, add_special_tokens=False)[:-1] == ids
        return out


def normalize_row(messages: list[dict]) -> list[dict]:
    """Nemotron-SFT-SWE rows: tool_calls is a JSON string of OpenAI tool calls whose arguments are JSON
    strings. Returns OpenAI-style messages with list tool_calls; raises ValueError on a malformed row."""
    out = []
    for m in messages:
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"role {role!r}")
        content = m.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise ValueError("non-string content")
        msg: dict[str, Any] = {"role": role, "content": content}
        if role == "assistant":
            rc = m.get("reasoning_content")
            if not isinstance(rc, str) or not rc.strip():
                raise ValueError("assistant without reasoning_content")
            if "<think>" in content or "</think>" in content:
                raise ValueError("think markup in content")
            msg["reasoning_content"] = rc
            calls = m.get("tool_calls")
            if isinstance(calls, str):
                calls = json.loads(calls) if calls.strip() else None
            if calls:
                norm = []
                for c in calls:
                    fn = c.get("function") or {}
                    name, args = fn.get("name"), fn.get("arguments")
                    if not isinstance(name, str) or not name:
                        raise ValueError("tool call without a name")
                    parsed = json.loads(args) if isinstance(args, str) else args
                    if not isinstance(parsed, dict):
                        raise ValueError("tool-call arguments are not a JSON object")
                    norm.append({"id": c.get("id") or f"call_{len(norm)}", "type": "function",
                                 "function": {"name": name, "arguments": args if isinstance(args, str)
                                              else json.dumps(args)}})
                msg["tool_calls"] = norm
        out.append(msg)
    if not any(m["role"] == "assistant" for m in out):
        raise ValueError("no assistant message")
    if out[-1]["role"] != "assistant":
        raise ValueError("does not end with an assistant message")
    return out
