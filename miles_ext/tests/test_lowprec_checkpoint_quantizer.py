"""miles_ext.lowprec.checkpoint_quantizer: syncs re-encode a DeepSeek V4.1-style checkpoint exactly.

The trainer starts from the BF16 dequantization of the release, so the first sync must hand the engine the
release's own values; after QAT-projected updates (--mxfp4-qat-routed-experts) it must hand over exactly the
experts the trainer computed with. CPU only.
"""

import pytest
import torch
from miles_ext.lowprec.checkpoint_quantizer import (
    MXFP4_GROUP_SIZE,
    checkpoint_layout,
    quantize_fp8_pow2_blocks,
    quantize_like_checkpoint,
    quantize_mxfp4_with_fp32_scales,
)
from safetensors.torch import save_file

from miles.utils.mxfp4 import dequantize_mxfp4, project_mxfp4

BLOCK = (32, 32)


def _dequant_fp8(qweight, scale, block=BLOCK):
    rows, cols = qweight.shape
    full = scale.repeat_interleave(block[0], 0).repeat_interleave(block[1], 1)[:rows, :cols]
    return qweight.float() * full


def _release_fp8(rows, cols, seed):
    """An FP8 tensor as a checkpoint stores it: E4M3 codes and a power-of-two scale per 32x32 tile."""
    g = torch.Generator().manual_seed(seed)
    codes = (torch.randn(rows, cols, generator=g) * 120).clamp(-448, 448).to(torch.float8_e4m3fn)
    tiles = (-(-rows // 32), -(-cols // 32))
    scale = torch.exp2(torch.randint(-14, -4, tiles, generator=g).float())
    return codes, scale


def _release_fp4(rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (rows, cols // 2), generator=g, dtype=torch.uint8)
    exponents = torch.randint(110, 125, (rows, cols // MXFP4_GROUP_SIZE), generator=g, dtype=torch.uint8)
    return packed, exponents


@pytest.mark.parametrize("shape", [(64, 96), (70, 50), (1, 32)])
def test_fp8_pow2_blocks_reencode_checkpoint_values_exactly(shape):
    codes, scale = _release_fp8(*shape, seed=0)
    bf16 = _dequant_fp8(codes, scale).to(torch.bfloat16)
    assert torch.equal(bf16.float(), _dequant_fp8(codes, scale)), "E4M3 x 2^k is exact in BF16"

    qweight, new_scale = quantize_fp8_pow2_blocks(bf16, BLOCK)

    assert qweight.dtype == torch.float8_e4m3fn and new_scale.dtype == torch.float32
    assert torch.equal(torch.log2(new_scale), torch.log2(new_scale).round()), "scales are powers of two"
    assert torch.equal(_dequant_fp8(qweight, new_scale), bf16.float())


def test_fp8_pow2_blocks_keep_every_tile_within_range():
    weight = (torch.randn(96, 64) * 3).to(torch.bfloat16)
    qweight, scale = quantize_fp8_pow2_blocks(weight, BLOCK)
    assert qweight.float().abs().max() <= 448
    rel_err = (_dequant_fp8(qweight, scale) - weight.float()).abs() / weight.float().abs().clamp(min=1e-3)
    assert rel_err.median() < 0.07


def test_mxfp4_reencodes_checkpoint_values_exactly():
    packed, exponents = _release_fp4(64, 128, seed=1)
    bf16 = dequantize_mxfp4(packed, exponents, MXFP4_GROUP_SIZE)

    qweight, scale = quantize_mxfp4_with_fp32_scales(bf16, int8=True)

    assert qweight.dtype == torch.int8 and scale.dtype == torch.float32
    reencoded = dequantize_mxfp4(qweight.view(torch.uint8), (torch.log2(scale) + 127).to(torch.uint8), 32)
    assert torch.equal(reencoded, bf16)


def test_mxfp4_sync_delivers_the_qat_projected_experts():
    weight = (torch.randn(32, 64) * 0.02).to(torch.bfloat16)
    trained = project_mxfp4(weight, MXFP4_GROUP_SIZE)  # what --mxfp4-qat-routed-experts leaves in the param

    qweight, scale = quantize_mxfp4_with_fp32_scales(trained, int8=False)

    served = dequantize_mxfp4(qweight, (torch.log2(scale) + 127).to(torch.uint8), MXFP4_GROUP_SIZE)
    assert torch.equal(served, trained)


@pytest.fixture
def v41_like_checkpoint(tmp_path):
    fp8, fp8_scale = _release_fp8(64, 96, seed=2)
    wo_a, wo_a_scale = _release_fp8(32, 64, seed=3)
    experts, exponents = _release_fp4(64, 128, seed=4)
    tensors = {
        "layers.0.attn.wq_a.weight": fp8,
        "layers.0.attn.wq_a.scale": fp8_scale,
        "layers.0.attn.wo_a.weight": wo_a,
        "layers.0.attn.wo_a.scale": wo_a_scale,
        "layers.0.ffn.experts.0.w1.weight": experts.view(torch.int8),
        "layers.0.ffn.experts.0.w1.scale": exponents.view(torch.int8),
        "layers.0.attn.indexer.wk.weight": torch.zeros(16, 32, dtype=torch.bfloat16),
    }
    save_file(tensors, str(tmp_path / "model-00001.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {' + ", ".join(f'"{n}": "model-00001.safetensors"' for n in tensors) + "}}"
    )
    checkpoint_layout.cache_clear()
    return tmp_path, tensors


def _sync(checkpoint, named):
    args = type("Args", (), {"hf_checkpoint": str(checkpoint)})()
    return dict(quantize_like_checkpoint(args, "module.module.x", named, {"weight_block_size": [32, 32]}))


def test_each_tensor_takes_the_format_the_checkpoint_stores(v41_like_checkpoint):
    checkpoint, release = v41_like_checkpoint
    dense = _dequant_fp8(release["layers.0.attn.wq_a.weight"], release["layers.0.attn.wq_a.scale"])
    expert = dequantize_mxfp4(
        release["layers.0.ffn.experts.0.w1.weight"].view(torch.uint8),
        release["layers.0.ffn.experts.0.w1.scale"].view(torch.uint8),
        32,
    )
    wo_a = torch.randn(32, 64, dtype=torch.bfloat16)
    wk = torch.randn(16, 32, dtype=torch.bfloat16)

    sent = _sync(
        checkpoint,
        [
            ("layers.0.attn.wq_a.weight", dense.to(torch.bfloat16)),
            ("layers.0.ffn.experts.0.w1.weight", expert),
            ("layers.0.attn.wo_a.weight", wo_a),
            ("layers.0.attn.indexer.wk.weight", wk),
        ],
    )

    assert torch.equal(_dequant_fp8(sent["layers.0.attn.wq_a.weight"], sent["layers.0.attn.wq_a.scale"]), dense)
    assert sent["layers.0.ffn.experts.0.w1.weight"].dtype == torch.int8
    assert sent["layers.0.ffn.experts.0.w1.scale"].shape == (64, 4)
    assert sent["layers.0.attn.wo_a.weight"] is wo_a and "layers.0.attn.wo_a.scale" not in sent
    assert sent["layers.0.attn.indexer.wk.weight"] is wk and "layers.0.attn.indexer.wk.scale" not in sent


def test_a_name_the_checkpoint_lacks_fails_loudly(v41_like_checkpoint):
    checkpoint, _ = v41_like_checkpoint
    with pytest.raises(KeyError, match="not in the rollout checkpoint"):
        _sync(checkpoint, [("layers.0.attn.wq_b.weight", torch.zeros(64, 96, dtype=torch.bfloat16))])


def test_a_shape_the_checkpoint_does_not_store_fails_loudly(v41_like_checkpoint):
    checkpoint, _ = v41_like_checkpoint
    with pytest.raises(AssertionError, match="rollout checkpoint stores"):
        _sync(checkpoint, [("layers.0.attn.wq_a.weight", torch.zeros(64, 128, dtype=torch.bfloat16))])


def test_a_packed_tensor_without_scales_fails_loudly(v41_like_checkpoint):
    checkpoint, release = v41_like_checkpoint
    tensors = {**release, "layers.0.ffn.experts.1.w1.weight": release["layers.0.ffn.experts.0.w1.weight"].clone()}
    save_file(tensors, str(checkpoint / "model-00001.safetensors"))
    (checkpoint / "model.safetensors.index.json").write_text(
        '{"weight_map": {' + ", ".join(f'"{n}": "model-00001.safetensors"' for n in tensors) + "}}"
    )
    checkpoint_layout.cache_clear()
    with pytest.raises(KeyError, match="has no layers.0.ffn.experts.1.w1.scale"):
        _sync(checkpoint, [("layers.0.ffn.experts.1.w1.weight", torch.zeros(64, 128, dtype=torch.bfloat16))])


def test_a_tensor_sent_unconverted_still_has_its_shape_checked(v41_like_checkpoint):
    checkpoint, _ = v41_like_checkpoint
    with pytest.raises(AssertionError, match="rollout checkpoint stores"):
        _sync(checkpoint, [("layers.0.attn.indexer.wk.weight", torch.zeros(16, 64, dtype=torch.bfloat16))])
