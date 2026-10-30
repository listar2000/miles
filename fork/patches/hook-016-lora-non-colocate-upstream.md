# hook-016: native LoRA syncs adapters to non-colocated engines (upstream's unmerged branch)

## Why
Upstream native LoRA (#1792) refuses adapter serving unless the engines are colocated
(`validate_native_lora_args`), and `--check-lora-weight-equal` records the adapter checksums on
rank 0 only, asserting that the weight iterator gathers PP stages. Our V4.1 LoRA recipes run the
trainer and the rollout engines on separate nodes. hook-011 (PP 1 only) and hook-012 (the checksum
assert also accepts PP 1) let them start.

## Decision
Carry upstream's own fix instead of ours: branch `yueming/native-lora-non-colocate` of
radixark/miles, commits `1fe8a1aaba` (drop the colocate-only guard; adapters already export
through the PP-local stream broadcast carries for Bridge LoRA) and `5ced68ee7e` (every sender
records its adapter shard's checksums; rank 0 gathers their union). The code is upstream's, so
the next sync after the branch merges resolves to upstream's side. It also allows PP > 1, which
our recipes still refuse until a LoRA run here uses it.

Left out: the branch's doc page edit (`docs/advanced/lora.md`) and its stale TODO removal in
`hf_weight_iterator.py` (both would be fork edits to upstream files for no behavior), and
`50a4baeeae` (resends the base under `--check-weight-update-equal`, which no LoRA recipe sets).

Rejected: keeping hook-011 and hook-012 (two local hooks for what upstream already wrote).

## How it is applied
`miles/backends/megatron_utils/lora/native/registry.py` (guard removed),
`miles/backends/training_utils/weight_update/updater.py` (`_gather_lora_checksums`) and the
accept case in `tests/fast/backends/megatron_utils/lora/native/test_specs.py`, each marked
`[fork hook-016]`. Verify: a split LoRA run with `--check-lora-weight-equal` passes its adapter
syncs (`[LORA-CHECK]` equal, engine weight version +1 per sync). On a sync that brings the branch
in, take upstream's side and add `Retires: hook-016`.

## Remove when
`yueming/native-lora-non-colocate` merges into a Miles release we sync to; upstream status: not
merged into radixark/miles main `65feb4a61` (checked 2026-10-29).
