"""DeepSeek-V4.1 native LoRA (ext-030): adapters, the merged weights the engines receive, and target resolution."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from miles_ext.lora import dsv41
from miles_ext.lowprec.checkpoint_quantizer import quantize_fp8_pow2_blocks

from miles.backends.megatron_utils.lora.native.config import LoRAConfig
from miles.backends.megatron_utils.lora.native.merge import merge_lora_into_weights
from miles.backends.megatron_utils.lora.native.registry import resolve_adapter_targets
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext
from miles.utils.lora.hf_lora_targets import resolve_hf_lora_targets

HIDDEN, Q_RANK, HEAD_DIM, HEADS, O_RANK, FFN = 16, 12, 8, 2, 8, 8
CONFIG = {"model_type": dsv41.MODEL_TYPE, "n_shared_experts": 1}


class _Linear(nn.Module):
    def __init__(self, out_features, in_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

    def forward(self, x):
        return x @ self.weight.t(), None


class _Naming:
    def block_prefix(self, layer, host_weight, leaves):
        in_shared_expert = any(parameter is host_weight for parameter in layer.shared.parameters())
        return "layers.0.ffn.shared_experts." if in_shared_expert else "layers.0.attn."


def _layer(targets=None, kv_out=None):
    """A V4.1 layer's adapted linears with the spec attached; ``targets`` None adapts all of them."""
    layer = nn.Module()
    layer.attn = nn.Module()
    layer.attn.linear_q_down_proj = _Linear(Q_RANK, HIDDEN)  # TE does not keep parallel_mode="duplicated"
    layer.attn.linear_q_up_proj = _Linear(HEADS * HEAD_DIM, Q_RANK)
    layer.attn.linear_kv_proj = _Linear(HEAD_DIM if kv_out is None else kv_out, HIDDEN)
    layer.attn.linear_proj = _Linear(HIDDEN, HEADS * O_RANK)
    layer.shared = nn.Module()
    layer.shared.linear_fc1 = _Linear(2 * FFN, HIDDEN)
    layer.shared.linear_fc2 = _Linear(HIDDEN, FFN)
    context = AttachContext(
        lora=LoRAConfig(rank=4, alpha=8, dropout=0.0, target_modules=targets),
        transformer_config=SimpleNamespace(
            hidden_size=HIDDEN, q_lora_rank=Q_RANK, kv_lora_rank=HEAD_DIM, sequence_parallel=False
        ),
        tp_size=1,
        tp_rank=0,
        naming=_Naming(),
    )
    count = dsv41.ARCH_SPEC.attention.attach(layer.attn, layer, context)
    count += dsv41.ARCH_SPEC.mlp.attach(layer.shared, layer, context)
    return layer, count


def _hosts(layer):
    return {name: module for name, module in layer.named_modules() if isinstance(module, _Linear)}


def _merged(layer):
    return merge_lora_into_weights([layer], {n: p for n, p in layer.named_parameters() if "lora" not in n})


def _large_adapters(layer):
    """Adapters whose merged delta is of the order of the weights, so it shows on the FP8 grid."""
    torch.manual_seed(0)
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            if "lora" in name:
                parameter.normal_(std=0.05)


def test_zero_init_is_identity():
    layer, count = _layer()
    assert count == 6  # wq_a, wq_b, wkv, wo_b, shared w1/w3, shared w2
    for name, host in _hosts(layer).items():
        x = torch.randn(3, host.weight.shape[1])
        assert torch.equal(host(x)[0], x @ host.weight.t()), name
        assert torch.equal(_merged(layer)[f"{name}.weight"], host.weight), name


def test_a_tp_sharded_wkv_is_refused():
    with pytest.raises(AssertionError, match="wkv must be replicated"):
        _layer(kv_out=HEAD_DIM // 2)


def test_merged_weight_reproduces_the_adapter_forward():
    layer, _ = _layer()
    _large_adapters(layer)
    merged = _merged(layer)
    for name, host in _hosts(layer).items():
        x = torch.randn(3, host.weight.shape[1])
        torch.testing.assert_close(host(x)[0], x @ merged[f"{name}.weight"].t(), rtol=1e-5, atol=1e-5, msg=name)


def test_sync_changes_exactly_the_adapted_tensors_on_the_fp8_grid():
    layer, _ = _layer(targets=("layers.*.attn.wq_b", "layers.*.ffn.shared_experts.w1"))
    with torch.no_grad():  # the trainer's BF16 copy is decoded from the FP8 release, so it sits on the grid
        for host in _hosts(layer).values():
            q, scale = quantize_fp8_pow2_blocks(host.weight, (4, 4))
            host.weight.copy_((q.float().view(-1, 4, q.shape[1] // 4, 4) * scale[:, None, :, None]).view_as(q))
    _large_adapters(layer)
    merged = _merged(layer)
    changed = {}
    for name, host in _hosts(layer).items():
        base, synced = (
            quantize_fp8_pow2_blocks(w, (4, 4))[0].view(torch.uint8) for w in (host.weight, merged[f"{name}.weight"])
        )
        changed[name] = (base != synced).float().mean().item()
    fc1_rows = slice(0, FFN)  # w1 is the gate half of the fused FC1; w3 stays the base
    fc1 = merged["shared.linear_fc1.weight"]
    assert torch.equal(fc1[FFN:], layer.shared.linear_fc1.weight[FFN:])
    assert not torch.equal(fc1[fc1_rows], layer.shared.linear_fc1.weight[fc1_rows])
    assert changed["attn.linear_q_up_proj"] > 0.5 and changed["shared.linear_fc1"] > 0.25, changed
    assert all(changed[n] == 0 for n in changed if n not in ("attn.linear_q_up_proj", "shared.linear_fc1")), changed


def test_targets_default_to_attention_and_the_shared_expert(dsv41_lora_spec):
    targets = resolve_hf_lora_targets(CONFIG)
    assert targets == [
        "layers.*.attn.wq_a",
        "layers.*.attn.wq_b",
        "layers.*.attn.wkv",
        "layers.*.attn.wo_b",
        "layers.*.ffn.shared_experts.w1",
        "layers.*.ffn.shared_experts.w3",
        "layers.*.ffn.shared_experts.w2",
    ]
    kwargs = dict(hf_modules=[], lora_type="lora", experts_shared_outer_loras=False)
    assert resolve_adapter_targets(CONFIG, targets, **kwargs) == targets
    with pytest.raises(AssertionError, match="does not implement adapters"):
        resolve_adapter_targets(CONFIG, ["layers.*.ffn.experts.*.w1"], **kwargs)
    with pytest.raises(AssertionError, match="does not implement adapters"):
        resolve_adapter_targets(CONFIG, ["layers.*.attn.wo_a"], **kwargs)
