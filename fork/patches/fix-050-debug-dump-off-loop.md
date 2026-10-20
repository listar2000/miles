# fix-050: the debug rollout dump runs after the capture, in a thread, in a GIL-friendly format

## Why
`--save-debug-rollout-data` dumps every training batch from `RolloutExecutor` on the rollout event loop. A
DeepSeek V4.1 batch with recorded expert routing (R3) is large: four batch dumps of a 3-node overfit run were
19-27 GB each. Each dump froze
the loop for 189 s, 223 s and 418 s (no trial progress, collects backing up). Recipes always pass the flag.

## Decision
- The dump moves from `_generate_rollout_data` to `get()`, after the output snapshot's capture. Upstream keeps no
  await between a batch arriving and its capture. The dump runs under `asyncio.to_thread` and is awaited, so `get`
  returns once the file exists, as before, but the loop serves trials meanwhile. Replayed batches are not dumped
  again, as before.
- A thread alone is not enough. `torch.save`'s zip format builds its records holding the GIL. A CPU probe
  (Miles' `save_debug_rollout_data` on 16 V4.1-size samples, 3.1 GB, with a 10 ms asyncio ticker) measured:

  | Dump | Wall | Longest loop pause |
  | --- | --- | --- |
  | inline, zip format (before) | 26.4 s | 26.4 s |
  | thread, zip format | 21.0 s | 5.8 s |
  | thread, legacy format, pickle protocol 5 (this fix) | 4.7 s | 0.14 s |

  The legacy format with protocol 5 lets numpy write its buffers straight to the file. `torch.load` reads it
  (the probe read it back equal, in 1.0 s against 8.9 s). It cannot be `mmap`ed; nothing maps rollout dumps.
- The dump still runs before the conversion, so its content is unchanged. One side effect: the captured snapshot now
  keeps `metadata["messages"]` (the trajectory dump pops it after the capture), but only with
  `--save-debug-trajectory-data`.
- Not done here: the batch conversion and DP split (80-176 s per batch in that run) stay on the loop. An upstream test
  asserts that the conversion runs on the caller loop, so moving it is a separate decision.

## How it is applied
`miles/ray/rollout/rollout_executor.py` (`get`, `_generate_rollout_data`) and `miles/ray/rollout/debug_data.py`
(`torch.save` arguments), marked `[fork fix-050]`. Verify with
`tests/fast/ray/rollout/test_rollout_executor.py::TestOutputSnapshotReplay::test_the_debug_dump_runs_after_the_capture_and_off_the_loop`
and `test_debug_data.py::TestRoundTrip::test_train_path_round_trip`. On a sync, keep the dump after `capture` and
outside `_generate_rollout_data`.

## Remove when
Upstream writes the debug dump off the rollout loop. Upstream status: not filed.
