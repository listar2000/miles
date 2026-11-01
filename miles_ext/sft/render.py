"""Render SFT steps into samples with session server v2's attach rule or native-template prefix matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from miles_ext.sft.record import SftRecord, Step

from miles.rollout.session.types import SessionRecord
from miles.rollout.session.v2.postprocessor_hub import default_postprocess
from miles.rollout.session.v2.session_state import _render_token_ids
from miles.rollout.session.v2.tree_trajectory import SessionTree, TrajectoryNode
from miles.utils.chat_template_utils import template
from miles.utils.chat_template_utils.tito_tokenizer import (
    DeepSeekV41TITOTokenizer,
    QwenNextTITOTokenizer,
    TITOTokenizer,
    extract_template_args,
    get_tito_tokenizer,
    resolve_fixed_chat_template,
)
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

ThinkingRetention = Literal["all", "template"]
THINKING_RETENTION: tuple[str, ...] = ("all", "template")
# TITO families whose tokenize_completion is checked against session-server renders (test_sft_render*.py).
SUPPORTED_FAMILIES: tuple[type[TITOTokenizer], ...] = (QwenNextTITOTokenizer, DeepSeekV41TITOTokenizer)

# Placeholder record for tree nodes: SFT steps have no HTTP exchange behind them.
_NO_RECORD = SessionRecord(timestamp=0.0, method="POST", path="/sft", request={}, response={}, status_code=200)
# Message keys that only the SFT record uses; templates never see them.
_RECORD_ONLY_KEYS = ("completion_token_ids",)


@dataclass(frozen=True)
class RenderedSample:
    """One training sample: all token ids, a loss mask over all of them, and the trained steps."""

    token_ids: list[int]
    loss_mask: list[int]
    trained_steps: tuple[int, ...]


@dataclass
class _StepNode:
    step: Step
    index: int
    node: TrajectoryNode


class SftRenderer:
    def __init__(
        self,
        tito: TITOTokenizer,
        *,
        thinking_retention: ThinkingRetention = "all",
        native_tokenizer: Any = None,
        native_template_kwargs: dict[str, Any] | None = None,
    ):
        if type(tito) not in SUPPORTED_FAMILIES:
            names = ", ".join(c.__name__ for c in SUPPORTED_FAMILIES)
            raise ValueError(f"SFT rendering supports {names}; got {type(tito).__name__}")
        if thinking_retention not in THINKING_RETENTION:
            raise ValueError(f"thinking_retention must be one of {THINKING_RETENTION}, got {thinking_retention!r}")
        if thinking_retention == "template" and native_tokenizer is None:
            raise ValueError("thinking_retention='template' renders with the native template: pass native_tokenizer")
        self.tito = tito
        self.thinking_retention = thinking_retention
        self.native_tokenizer = native_tokenizer
        # Launch-level defaults for the native render; a record's chat_template_kwargs override them.
        self.native_template_kwargs = dict(native_template_kwargs or {})

    def render(self, record: SftRecord) -> list[RenderedSample]:
        """All samples of *record*, in the order of their leaf steps."""
        return self.render_within(record, max_tokens=None)[0]

    def render_within(self, record: SftRecord, *, max_tokens: int | None) -> tuple[list[RenderedSample], int]:
        """The samples of at most *max_tokens* tokens, and how many longer ones were dropped.

        Longer leaves are dropped before reply ownership, so a reply they share is trained in a kept one."""
        template_args = self._template_args(record)
        tree = SessionTree()
        nodes: list[_StepNode] = []
        for index, step in enumerate(record.steps):
            context = [_render_view(m) for m in record.context_messages(step)]
            reply = record.messages[step.completion]
            if self.thinking_retention == "all":
                parent, prompt, delta = self._attach_session(tree, context, template_args)
            else:
                parent, prompt, delta = self._attach_template(tree, context, self._render_kwargs(record))
            completion = reply.get("completion_token_ids") or self.tito.tokenize_completion(
                _render_view(reply), template_args=template_args
            )
            node = tree.create_node(
                parent,
                delta_messages=delta + [_render_view(reply)],
                token_ids=prompt + list(completion),
                completion_span=(len(prompt), len(prompt) + len(completion)),
                committed_at=0.0,
                response_id=f"step-{index}",
                record=_NO_RECORD,
                finish_reason="stop",
            )
            nodes.append(_StepNode(step=step, index=index, node=node))
        return _leaf_samples(tree, nodes, max_tokens)

    def _render_kwargs(self, record: SftRecord) -> dict[str, Any]:
        """The record's request kwargs and tools; in template mode over the native launch defaults."""
        kwargs = dict(record.chat_template_kwargs)
        if self.thinking_retention == "template":
            kwargs = {**self.native_template_kwargs, **kwargs}
        kwargs["tools"] = record.tools or kwargs.get("tools")
        return kwargs

    def _template_args(self, record: SftRecord) -> dict[str, Any]:
        """The record's render settings resolved by the family's request rules, as for a served request."""
        kwargs = self._render_kwargs(record)
        request = {"tools": kwargs.pop("tools"), "chat_template_kwargs": kwargs}
        return extract_template_args(self.tito.resolve_request_args(request, turn_args=None))

    def _attach_session(
        self, tree: SessionTree, context: list[dict[str, Any]], template_args: dict[str, Any]
    ) -> tuple[TrajectoryNode | None, list[int], list[dict[str, Any]]]:
        """Attach where the session server would: the deepest step whose messages prefix this context."""
        parent = tree.find_attach_point(context).node
        prompt = _render_token_ids(parent, context, template_args=template_args, tito_tokenizer=self.tito)
        matched = len(parent.path_messages()) if parent is not None else 0
        return parent, list(prompt), context[matched:]

    def _attach_template(
        self, tree: SessionTree, context: list[dict[str, Any]], native_args: dict[str, Any]
    ) -> tuple[TrajectoryNode | None, list[int], list[dict[str, Any]]]:
        """Render the context whole, as a stateless server does; attach under the deepest step whose
        tokens prefix the render. Template-mode nodes keep whole contexts: the tree is matched by tokens."""
        prompt = list(
            template.apply_chat_template(
                context, tokenizer=self.native_tokenizer, add_generation_prompt=True, tokenize=True, **native_args
            )
        )
        prefixes = [n for n in tree.nodes if prompt[: len(n.token_ids)] == n.token_ids]
        parent = max(prefixes, key=lambda n: (len(n.token_ids), n.seq), default=None)
        return parent, prompt, context


def _render_view(message: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in message.items() if k not in _RECORD_ONLY_KEYS}


def _leaf_samples(
    tree: SessionTree, nodes: list[_StepNode], max_tokens: int | None
) -> tuple[list[RenderedSample], int]:
    """One sample per kept leaf; each reply is trained once, in the earliest kept leaf that holds it."""
    by_seq = {item.node.seq: item for item in nodes}
    samples = []
    too_long = 0
    for leaf in tree.leaves():
        if max_tokens is not None and len(leaf.token_ids) > max_tokens:
            too_long += 1
            continue
        path = leaf.path_nodes()
        response_start = path[0].completion_span[0]
        mask = [0] * len(leaf.token_ids)
        for node in path:
            if by_seq[node.seq].step.train:
                start, end = node.completion_span
                mask[start:end] = [1] * (end - start)
        samples.append(
            Sample(
                tokens=list(leaf.token_ids),
                response_length=len(leaf.token_ids) - response_start,
                loss_mask=mask[response_start:],
                metadata={"leaf": {"node_id": leaf.seq, "path_node_ids": [n.seq for n in path]}},
            )
        )
    tree_nodes = [{"id": n.seq, "completion_span": list(n.completion_span)} for n in tree.nodes]
    owned = default_postprocess(samples, {"tree": {"nodes": tree_nodes}})
    out = []
    for sample in owned:
        prompt_length = len(sample.tokens) - sample.response_length
        full_mask = [0] * prompt_length + list(sample.loss_mask)
        path_ids = sample.metadata["leaf"]["path_node_ids"]
        trained = tuple(
            by_seq[seq].index
            for seq in path_ids
            if any(full_mask[slice(*tree.nodes[seq].completion_span)]) and by_seq[seq].step.train
        )
        out.append(RenderedSample(token_ids=list(sample.tokens), loss_mask=full_mask, trained_steps=trained))
    return out, too_long


def load_renderer(
    hf_checkpoint: str,
    tito_model: str,
    *,
    thinking_retention: ThinkingRetention = "all",
    chat_template_kwargs: dict[str, Any] | None = None,
) -> SftRenderer:
    """The renderer for a checkpoint, with the tokenizer and template the session server loads
    for ``--tito-model`` (``miles/utils/arguments.py`` and ``miles/rollout/session/sessions.py``)."""
    template_path, fixed_kwargs = resolve_fixed_chat_template(tito_model)
    tokenizer = load_tokenizer(hf_checkpoint, chat_template_path=template_path, trust_remote_code=True)
    tito = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=tito_model,
        chat_template_kwargs={**(chat_template_kwargs or {}), **fixed_kwargs},
    )
    native = load_tokenizer(hf_checkpoint, trust_remote_code=True) if thinking_retention == "template" else None
    return SftRenderer(
        tito,
        thinking_retention=thinking_retention,
        native_tokenizer=native,
        native_template_kwargs=chat_template_kwargs,
    )
