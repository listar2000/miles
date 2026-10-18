# fix-021: a fresh NVMe-streamed optimizer builds its fp32 mains at the first step

## Why
With `--stream-optimizer-state-to-disk` and no optimizer state to resume, `reload_model_params()` wrote every fp32
main to the node-local disk right after the weight load (fix-010 made that the only write). The first `step()` then
read them straight back. For DeepSeek-V4.1-Flash on 16 GPUs this is 126.6 GiB per rank written at startup (9-12 min
of the ~17 min trainer init on a 16-GPU run, ~0.2 GB/s per rank) and read again in step 0, while every GPU of the
run waits.

## Decision
On that path the store only records that the mains equal the model params (`defer_main_to_model_params`). The first
`step()` fills each bucket's mains from the model-param shard on the GPU instead of reading them, then writes them
with the moments as every step does. Nothing writes the model params between the load and that step (the step's
copy-back comes after), so the values are exactly the ones fix-010 wrote. A save before the first step writes them
first (`_ensure_main`). Resume (`load_from`) and a load that passes a state dict (`refresh_main_from_model_params`)
are unchanged. Alternatives that lost:
- A faster startup write (no per-bucket fdatasync, O_DIRECT): still a full write and a read nobody needs.
- Keeping the mains in host RAM until the step: 126.6 GiB per rank, which the 256k checkpoint offload already needs.

## How it is applied
`miles_plugins/optimizers/nvme_stream.py`, every spot marked `[fork fix-021]`. The startup log line is now
"NVMe optimizer main-param initialization: deferred to the first step", and step 0's "NVMe streaming step" line reports
read 0.0 GB. Verify: `tests/fast/optimizers/test_nvme_stream_deferred_main_init.py` (step 0 reads nothing and sees the same
mains as a written-then-read store, a save before step 0 writes them); on GPU, the deferred line at load and step 0's
"read 0.0 GB" show the wiring. On a sync, keep every
writer of the mains clearing `_main_from_model`.

## Remove when
Upstream `nvme_stream` stops writing fresh mains at load. Status 2026-10-18: not filed.
