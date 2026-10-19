"""DeepSeek-V4.1 for native LoRA: ``--custom-lora-spec-path miles_ext.lora.dsv41``.

| HF target (``layers.*.``) | Megatron host | Layout |
| --- | --- | --- |
| ``attn.wq_a``, ``attn.wkv`` | duplicated ``linear_q_down_proj``, ``linear_kv_proj`` | replicated |
| ``attn.wq_b`` | ``linear_q_up_proj`` | column |
| ``attn.wo_b`` | ``linear_proj`` | row |
| ``ffn.shared_experts.w1``, ``w3`` / ``w2`` | shared expert ``linear_fc1`` / ``linear_fc2`` | column / row |

"""

import re
from dataclasses import dataclass

import torch.nn as nn

from miles.backends.megatron_utils.lora.native.modules.linear import LoRASplitFC1
from miles.backends.megatron_utils.lora.native.spec.base import (
    AttachContext,
    LoRAArchSpec,
    ProjectionSpec,
    ShardLayout,
)
from miles.backends.megatron_utils.lora.native.spec.layout import (
    AttentionSpecBase,
    FusedAttach,
    LayoutSpec,
    ModuleLayout,
    ProjectionBinding,
    config_dim,
    hidden_size,
    host_in_local,
)

MODEL_TYPE = "deepseek_v41"
HF_LAYER_PREFIX = "layers.*"
HF_UNEMBED = "head"
# no routed experts: the engines serve them in MXFP4, where small merged deltas round away
_ATTENTION = ("attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b")
_SHARED_EXPERT = ("ffn.shared_experts.w1", "ffn.shared_experts.w3", "ffn.shared_experts.w2")


def hf_lora_groups(config: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return _ATTENTION, _SHARED_EXPERT if config.get("n_shared_experts") else ()


def _host_out_local(host_attr: str):
    def resolve(module: nn.Module, _context: AttachContext) -> int:
        return getattr(module, host_attr).weight.shape[0]

    return resolve


def _replicated_guard(host: nn.Module, _context: AttachContext, projection: ProjectionSpec, full_out: int) -> None:
    """Upstream's MLA check: TE does not keep ``parallel_mode='duplicated'``, so a full-width weight also counts."""
    assert getattr(host, "parallel_mode", None) == "duplicated" or host.weight.shape[0] == full_out, (
        f"{projection.hf_leaf} must be replicated across TP; this build shards it "
        f"({tuple(host.weight.shape)} vs full out {full_out})"
    )


def _replicated(leaf: str, host_attr: str, full_out: str) -> ProjectionBinding:
    return ProjectionBinding(
        projection=ProjectionSpec(leaf, leaf, ShardLayout.REPLICATED),
        module_attr=host_attr,
        in_dim=hidden_size,
        out_dim=config_dim(full_out),
        adapter_attr=f"lora_{leaf}_adapter",
        guard=_replicated_guard,
    )


class DeepSeekV41AttentionSpec(AttentionSpecBase):
    name = "deepseek_v41_attention"
    layout = ModuleLayout(
        name="deepseek_v41_attention",
        hf_block_prefix="attn.",
        full_sequence=True,  # the block gathers the sequence-parallel input and runs its linears without SP
        singles=(
            _replicated("wq_a", "linear_q_down_proj", "q_lora_rank"),
            ProjectionBinding(
                projection=ProjectionSpec("wq_b", "wq_b", ShardLayout.COLUMN),
                module_attr="linear_q_up_proj",
                in_dim=host_in_local("linear_q_up_proj"),
                out_dim=_host_out_local("linear_q_up_proj"),
                adapter_attr="lora_wq_b_adapter",
            ),
            _replicated("wkv", "linear_kv_proj", "kv_lora_rank"),
            ProjectionBinding(
                projection=ProjectionSpec("wo_b", "wo_b", ShardLayout.ROW),
                module_attr="linear_proj",
                in_dim=host_in_local("linear_proj"),
                out_dim=hidden_size,
                adapter_attr="lora_wo_b_adapter",
            ),
        ),
    )


def _fc1_half_local(mlp: nn.Module, _context: AttachContext) -> int:
    return mlp.linear_fc1.weight.shape[0] // 2


def _build_split_fc1(mlp: nn.Module, hf_prefix: str, context: AttachContext, active) -> LoRASplitFC1:
    return LoRASplitFC1(
        hf_prefix=hf_prefix,
        reference=mlp.linear_fc1.weight,
        context=context,
        projections=active,
        inter_local=_fc1_half_local(mlp, context),
    )


class DeepSeekV41SharedExpertSpec(LayoutSpec):
    """The shared expert's fused ``[w1; w3]`` FC1 and its ``w2``."""

    name = "deepseek_v41_shared_expert"
    layout = ModuleLayout(
        name="deepseek_v41_shared_expert",
        present_when_attr="linear_fc1",
        fused=(
            FusedAttach(
                module_attr="linear_fc1",
                projections=(
                    ProjectionSpec("w1", "gate", ShardLayout.COLUMN),
                    ProjectionSpec("w3", "up", ShardLayout.COLUMN),
                ),
                adapter_attr="lora_fc1_adapter",
                build=_build_split_fc1,
            ),
        ),
        singles=(
            ProjectionBinding(
                projection=ProjectionSpec("w2", "down", ShardLayout.ROW),
                module_attr="linear_fc2",
                in_dim=_fc1_half_local,
                out_dim=hidden_size,
                adapter_attr="lora_fc2_adapter",
            ),
        ),
    )


@dataclass(frozen=True)
class DeepSeekV41ArchSpec(LoRAArchSpec):
    """Upstream's ``attaches`` expects HF ``mlp`` blocks; V4.1 names the shared expert ``ffn.shared_experts``."""

    def attaches(self, hf_module: str) -> bool:
        block, _, leaf = hf_module.rpartition(".")
        if leaf in self.attention.supported_targets:
            return re.search(r"(?:^|\.)layers\.(?:\d+|\*)\.attn$", block) is not None
        if leaf in self.mlp.supported_targets:
            return re.search(r"(?:^|\.)layers\.(?:\d+|\*)\.ffn\.shared_experts$", block) is not None
        return False


ARCH_SPEC = DeepSeekV41ArchSpec(
    name="deepseek_v41", attention=DeepSeekV41AttentionSpec(), mlp=DeepSeekV41SharedExpertSpec()
)
