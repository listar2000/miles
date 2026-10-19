# fix-022: the trainer can initialize while the rollout side starts its engines

## Why
`train_async.py` awaited `create_rollout_components` (router, session servers, SGLang engines: 9.0 min in
the 2-node DeepSeek-V4.1 RL run of 2026-10-18, https://wandb.ai/mobiuslab/miles-dsv41/runs/l0huvjij) before it called `create_training_models`, so the trainer's model build and weight load (about 5 min
on DeepSeek-V4.1-Flash, more without fix-021, PR #39) started only after the engines were healthy, and every GPU waited for
both in turn. The trainer needs nothing from the rollout side until it reports its parallel layout.

## Decision
`--trainer-init-with-rollout` (default off) runs `create_rollout_components` and the new `init_training_models` together
and calls `connect_training_models` (the parallel-config handoff and the rollout data reload on resume) once both are
done. `create_training_models` is now those two calls in order, so the default path is unchanged. The flag needs
`--num-rollout`: with only `--num-epoch` the trainer's iteration count waits for the rollout side to count the
dataset, and it refuses `--colocate`, where an engine sizes its memory from what is free on GPUs the trainer is
loading onto. Both sides then read their checkpoints at the same time; four containers reading one Modal Volume
together each kept 2.4-5.3 GB/s (CPU read test, 2026-10-18), so sharing a volume costs nothing measurable. The existing
`MILES_SGLANG_DUMMY_LOAD=1` would still skip the engines' read, since the first weight update fills them.
Alternatives that lost:
- Gating only the trainer's weight read on engine readiness: hides about 1 min (actor spawn already overlaps).
- Starting the trainer first and the engines after it: the engine start is the longer of the two.

## How it is applied
`train_async.py`, `miles/ray/placement_group.py` and `miles/utils/arguments.py`, every spot marked `[fork fix-022]`.
Verify: `tests/fast/test_train_async.py::TestTrainerInitWithRollout` (the trainer init starts before the rollout side
is ready, and connects after both). On a sync, keep `create_training_models` equal to init + connect.

## Remove when
Upstream starts the trainer alongside the rollout side. Status 2026-10-18: not filed.
