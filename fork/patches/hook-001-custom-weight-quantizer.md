# hook-001: `--custom-weight-quantizer-path`, a checkpoint layout brings its own sync quantizer

## Why
`quantize_params` (`megatron_to_hf/processors/__init__.py`) picks one built-in quantizer per
checkpoint from its `quantization_config`. A rollout checkpoint whose layout none of them covers
has no way in. DeepSeek V4.1-Flash's release is one: the FP8 quantizer asserts on its 32x32 blocks
on Blackwell (`quant_weight_ue8m0` needs 128x128) and would send the FP4 routed experts as FP8.

## Decision
A function-path flag in the style of Miles' other `--custom-*-path` flags. When set, the function
`(args, megatron_name, converted_named_params, quantization_config)` owns every converted tensor of
the sync and returns the `(name, tensor)` list to send. Unset (the default), nothing changes.
Rejected: another built-in branch in the dispatch, which would need a model-specific upstream edit
per layout. First user: ext-007.

## How it is applied
`miles/utils/fork_arguments.py` (flag, hook-014) and `quantize_params` (the early return); both spots are marked.
On a sync, re-insert the few lines if upstream reshapes `quantize_params`.

## Remove when
Upstream offers an equivalent extension point, or syncs every layout we serve natively. Status:
not filed.
