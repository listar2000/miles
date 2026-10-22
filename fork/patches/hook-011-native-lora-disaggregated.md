# hook-011: native LoRA syncs adapters to non-colocated engines without pipeline parallelism

## Why
Upstream native LoRA (#1792) refuses adapter serving unless the engines are colocated
(`validate_native_lora_args`). The V4.1 LoRA overfit recipes run the trainer and the rollout engines on separate nodes, so
unmerged serving (hook-010, `lora.serving: adapter`) could not start.

## Decision
- Allow it when `--pipeline-model-parallel-size 1`. The raw exporter (`export_lora_hf_named`) gathers TP/EP
  and materializes the whole adapter on every rank, and the NCCL broadcast protocol already declares
  `supports_lora` and sends adapter tensors as `{lora_name}:{hf_key}` (the bridge-mode path upstream uses).
  With PP the adapter is PP-local and needs the PP gather upstream marks as untested, so PP stays refused.
- Rejected: bridge mode for V4.1 (no V4.1 bridge LoRA); colocating the V4.1 LoRA recipes (the recipe is disaggregated for memory).

## How it is applied
One condition in `miles/backends/megatron_utils/lora/native/registry.py`, `[fork hook-011]`, and its case in
`tests/fast/backends/megatron_utils/lora/native/test_specs.py`. Verify: an adapter-serving run on separate
rollout nodes reports `[LORA-CHECK]` checksums equal and the engine weight version +1 per sync.

## Remove when
Upstream lifts the colocate-only guard for native LoRA.
