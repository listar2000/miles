# hook-013: `--custom-weight-transfer-protocol-path` picks the weight-transfer protocol by import path

## Why
`get_weight_transfer_protocol` is a closed if/elif over `--update-weight-transfer-mode`. A split run
(ext-036) has no network path for NCCL between the trainer and the engine container, so its adapter
sync needs a protocol Miles does not ship (ext-037, through the results volume).

## Decision
- One flag and three inserted lines at the top of the factory: when the path is set, Miles loads it with
  `load_function` and calls it with `args`; every other mode is unchanged.
- The protocol keeps the updater, the exporter, the session frame and the LoRA checksums; only its own
  `connect` / `send_bucket` / `after_base_weights` differ, so a new transport needs no other edit.
- Rejected: a new `--update-weight-transfer-mode` choice in upstream code (the transport lives in
  `miles_ext`, a mode name would tie upstream to it); monkeypatching the factory from a
  `--custom-megatron-init-path` hook (a runtime patch of upstream code).

## How it is applied
`miles/utils/fork_arguments.py` (the flag, hook-014) and
`miles/backends/training_utils/weight_update/protocol.py` (the check), marked `[fork hook-013]`.
Verify: a split run logs `[split sync vN]` lines and passes `--check-lora-weight-equal`. On a sync, re-insert the three
lines at the top of `get_weight_transfer_protocol`.

## Remove when
Upstream takes a custom protocol path; upstream status: not filed.
