# fix-073: NVMe-streamed optimizer state in host memory up to a per-GPU budget, again (fix-045, for GLM-5.3-Flash)

## Why
fix-045 added `--stream-optimizer-state-host-gb`, and fix-060 removed it on 2026-10-27 because no run set it.
Full-parameter GLM-5.3-Flash on one B300:8 trainer node (ext-051, `glm53-flash-stak`) now needs it. Adam's
state there is ~2.6 TB per node (8 bytes per parameter with bf16 moments, 321 B parameters), against a 3 TiB
disk that Modal caps at ~2.6 GB/s per container. Streaming all of it through the disk reads and writes ~5 TB per
step. Most of it fits in host memory: a B300:8 node accepts up to 2,560 GiB.

## Decision
The same change as fix-045 (its page has the design and the measurements: byte-identical state, the slowest
rank's steady step 1,541 s on disk vs 1,064 s with 100 GiB/GPU in host memory on full-size V4.1 expert shapes).
The code is fix-045's `nvme_stream.py` and its test, restored from git. One difference: the flag is in
`miles/utils/fork_arguments.py` (hook-014), not in `arguments.py`. The recipe sets it with `optimizer.host_gib`.

## How it is applied
`miles_plugins/optimizers/nvme_stream.py`, `miles/utils/fork_arguments.py` and `miles/utils/arguments.py` (the Adam-only
check), every spot marked `[fork fix-073]`. The recipe key is `optimizer.host_gib` (`miles_ext/recipes/recipe.py`,
ext-051). Size it to what `cluster.memory_gib` leaves after Ray, the trainer processes and the HF load. Verify:
`tests/fast/optimizers/test_nvme_stream_host_tier.py` (needs a GPU), and the store's startup line on the
`glm53-flash-stak` run reporting how many GB are in host memory.

## Remove when
Upstream `nvme_stream` gains a host-memory tier; upstream status: not filed.
