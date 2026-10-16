"""Weight-sync quantizer that writes each tensor in the format the rollout checkpoint stores it.

Plugged in with ``--custom-weight-quantizer-path miles_ext.lowprec.checkpoint_quantizer.quantize_like_checkpoint``
(hook-001). The rollout engine loads ``--hf-checkpoint`` from disk and every sync must hand it the same
layout, so the checkpoint's safetensors headers are the single source of truth for each tensor:

- ``F8_E4M3`` weight: FP8 with power-of-two (UE8M0) block scales, block from
  ``quantization_config.weight_block_size`` (DeepSeek V4.1-Flash: 32x32, served as MXFP8 on Blackwell).
- ``I8``/``U8`` weight with a ``.scale``: MXFP4 (E2M1 nibbles, one E8M0 scale per 32 elements), the
  DeepSeek routed-expert layout, packed by ``miles.utils.mxfp4.quantize_mxfp4`` so its numerics match
  ``--mxfp4-qat-routed-experts``.
- anything else: sent as converted (BF16).

Scales are sent as fp32 powers of two: the engine's scale params are fp32 (it casts expert scales to E8M0
when it re-shuffles them), and fp32 travels over every transfer mode.
"""

import json
from functools import lru_cache
from pathlib import Path

import torch
from safetensors import safe_open

from miles.utils.mxfp4 import quantize_mxfp4

MXFP4_GROUP_SIZE = 32
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
_FP8_DTYPES = frozenset({"F8_E4M3"})
_FP4_DTYPES = frozenset({"I8", "U8"})
# The checkpoint stores DeepSeek V4's attention output projection wo_a in FP8, but the engine dequantizes it
# to BF16 at load (models/deepseek_v4.py, SGLANG_OPT_FP8_WO_A_GEMM=0, which online weight updates need) and
# its streaming loader drops a scale that arrives apart from its weight. Send what the engine holds.
_ENGINE_BF16_SUFFIXES = (".attn.wo_a.weight",)


def quantize_like_checkpoint(args, megatron_name, converted_named_params, quantization_config):
    layout = checkpoint_layout(args.hf_checkpoint)
    block = tuple(quantization_config["weight_block_size"]) if quantization_config else None
    out = []
    for name, param in converted_named_params:
        out += _quantize_one(name, param, layout, block)
    return out


def _quantize_one(name, param, layout, block):
    if name not in layout:
        raise KeyError(
            f"{name} (converted for the rollout engine) is not in the rollout checkpoint; the converter and "
            "--hf-checkpoint disagree on the model layout"
        )
    stored_shape, stored_dtype = layout[name]
    if name.endswith(_ENGINE_BF16_SUFFIXES):
        _check_shape(name, param, stored_shape)
        return [(name, param)]
    scale_name = name.removesuffix(".weight") + ".scale"
    if stored_dtype in _FP8_DTYPES:
        assert block is not None, f"{name} is FP8 in the checkpoint but quantization_config has no weight_block_size"
        qweight, scale = quantize_fp8_pow2_blocks(param, block)
    elif stored_dtype in _FP4_DTYPES:
        if scale_name not in layout:
            raise KeyError(f"{name} is {stored_dtype} in the rollout checkpoint but has no {scale_name}")
        qweight, scale = quantize_mxfp4_with_fp32_scales(param, int8=stored_dtype == "I8")
    else:
        _check_shape(name, param, stored_shape)
        return [(name, param)]
    _check_shape(name, qweight, stored_shape)
    _check_shape(scale_name, scale, layout[scale_name][0])
    return [(name, qweight), (scale_name, scale)]


def quantize_fp8_pow2_blocks(weight: torch.Tensor, block: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 E4M3 with one power-of-two scale per ``block`` tile (the smallest that keeps the tile within 448).

    A power-of-two scale only moves exponents, so a weight that is already FP8 on this grid (e.g. the BF16
    dequantization of the checkpoint) re-encodes to the same values.
    """
    assert weight.dim() == 2, f"FP8 block quantization expects a 2-D weight, got {tuple(weight.shape)}"
    rows, cols = weight.shape
    block_rows, block_cols = block
    padded = torch.zeros(
        -(-rows // block_rows) * block_rows,
        -(-cols // block_cols) * block_cols,
        dtype=weight.dtype,
        device=weight.device,
    )
    padded[:rows, :cols] = weight
    tiles = padded.view(padded.shape[0] // block_rows, block_rows, padded.shape[1] // block_cols, block_cols)
    # tiny, not DeepGEMM's 1e-4 floor: a raised floor would coarsen the grid of near-zero tiles.
    amax = tiles.abs().float().amax(dim=(1, 3), keepdim=True).clamp(min=torch.finfo(torch.float32).tiny)
    scale = _ceil_pow2(amax / _FP8_MAX)
    qweight = (tiles.float() / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    qweight = qweight.view_as(padded)[:rows, :cols].contiguous()
    return qweight, scale.view(tiles.shape[0], tiles.shape[2]).contiguous()


def quantize_mxfp4_with_fp32_scales(weight: torch.Tensor, *, int8: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP4 packed nibbles plus their E8M0 block scales decoded to fp32 (2^(e-127))."""
    packed, biased_exponent = quantize_mxfp4(weight, MXFP4_GROUP_SIZE)
    scale = torch.exp2(biased_exponent.float() - 127.0)
    return (packed.view(torch.int8) if int8 else packed), scale


def _ceil_pow2(x: torch.Tensor) -> torch.Tensor:
    """Smallest power of two >= x for positive normal fp32 x, exact (bit-level, as DeepGEMM's ceil_to_ue8m0)."""
    bits = x.float().contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return (exponent.clamp(1, 254) << 23).view(torch.float32)


def _check_shape(name, tensor, stored_shape):
    assert tuple(tensor.shape) == tuple(
        stored_shape
    ), f"{name}: sync would send shape {tuple(tensor.shape)}, the rollout checkpoint stores {tuple(stored_shape)}"


@lru_cache(maxsize=2)
def checkpoint_layout(hf_checkpoint: str) -> dict[str, tuple[tuple[int, ...], str]]:
    """{tensor name: (shape, safetensors dtype)} from the checkpoint's headers; no payload is read."""
    root = Path(hf_checkpoint)
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    names_by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        names_by_shard.setdefault(shard, []).append(name)
    layout = {}
    for shard, names in names_by_shard.items():
        with safe_open(root / shard, framework="pt", device="cpu") as tensors:
            for name in names:
                tensor_slice = tensors.get_slice(name)
                layout[name] = (tuple(tensor_slice.get_shape()), tensor_slice.get_dtype())
    return layout
