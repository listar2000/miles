"""Model presets: what a run needs to know about a model beyond its weights.

``tito_model`` is Miles' token-in/token-out template family; the two sglang
parsers are the ones Miles binds to that family
(miles/utils/chat_template_utils/tito_tokenizer.py). Miles v0.1.2 does not apply
them itself, so they are passed explicitly (RUN_LOG.md item 7).

``megatron_model_type`` names the model's Megatron definition,
``scripts/models/<megatron_model_type>.py`` (what Miles' ``load_model_args``
reads); a recipe that sets ``model.backend: megatron`` needs it.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Preset:
    hf_repo: str
    tito_model: str
    tool_call_parser: str
    reasoning_parser: str
    backend: str = "fsdp"
    megatron_model_type: str | None = None
    chat_template_kwargs: dict = field(default_factory=dict)
    # cluster shapes this preset has been run on: (gpu type, nodes, backend)
    shapes: tuple[tuple[str, int, str], ...] = ()

    @property
    def dir_name(self) -> str:
        return self.hf_repo.split("/")[-1]


PRESETS: dict[str, Preset] = {
    # Run 1 (overfit8-async-v1): FSDP, one H200:8 node, non-thinking
    "qwen3.5-9b": Preset(
        hf_repo="Qwen/Qwen3.5-9B",
        tito_model="qwen35",
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
        chat_template_kwargs={"enable_thinking": False},
        megatron_model_type="qwen3.5-9B",
        shapes=(("H200", 1, "fsdp"),),
    ),
    "qwen3-8b": Preset(
        hf_repo="Qwen/Qwen3-8B",
        tito_model="qwen3",
        tool_call_parser="qwen25",
        reasoning_parser="qwen3",
        chat_template_kwargs={"enable_thinking": False},
        megatron_model_type="qwen3-8B",
        shapes=(("H200", 1, "fsdp"),),
    ),
}
