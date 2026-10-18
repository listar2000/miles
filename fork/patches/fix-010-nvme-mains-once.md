# fix-010: NVMe-streamed fp32 mains written once, after the checkpoint load

## Why
With `--stream-optimizer-state-to-disk`, `setup_optimizer_state_streaming` wrote every fp32 main from the
freshly initialised model params before `load_checkpoint`, and the load then wrote them all again
(`reload_model_params`, reached through Megatron's `ChainedOptimizer` after an HF or `--no-load-optim` load) or
copied the saved bucket files over them (`load_from` on resume). The first pass was always overwritten. For
DeepSeek-V4.1-Flash on 16 GPUs it is 126.6 GB per rank, ~10 of the ~22 minutes of trainer init
(dsv41-tb4-overfit16-4 rank 0: 11:04:50-11:14:37Z; init elapsed 1305.6 s), paid on every launch and restart while
all the run's GPUs wait.

## Decision
The load path is the only writer. A `_main_ready` flag is set by the three writers
(`initialize_main_from_model_params`, `refresh_main_from_model_params`, `load_from`); `step()` and `save_to()`
call `_ensure_main()`, which fills the mains from the current model params (with a warning) if no load path did,
so no path can step or save from unwritten mains. Alternatives that lost:
- Faster first write (O_DIRECT, larger chunks): still a wasted pass.
- Skip the write only for HF loads: every load path overwrites it, and a per-path rule is easy to break.
Numerics are unchanged: the mains equal the loaded params, as before.

## How it is applied
`miles_plugins/optimizers/nvme_stream.py`, every spot marked `[fork fix-010]`. The "NVMe optimizer main-param
initialization" log line appears after "Load checkpoint ...": "wrote N GB" when a load path wrote the mains,
"deferred to the first step" on a fresh start (fix-021). Verify:
`tests/fast/optimizers/test_nvme_stream_deferred_main_init.py` (CPU: written exactly once, equal to the loaded
params; a resume via `load_from` writes nothing) and, on GPU, one such line per store after the load and no
"no checkpoint load filled the fp32 mains" warning. On a sync, keep the flag on every path that writes the mains.

## Remove when
Upstream `nvme_stream` stops initialising the mains before `load_checkpoint`. Status 2026-10-17: not filed.
