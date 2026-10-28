# fix-063: a fresh run starts rollout 0 as soon as its engines are ready

## Why
Rollout waited for the trainer's first weight sync (version 0), and the trainer sends it only after its model build
and weight load. On a fresh unmerged-LoRA run that sync carries nothing new: the adapter starts with lora_B = 0, so
the policy is the engines' own base. Ready engines still idled behind it, 6-25 min on split DeepSeek-V4.1 runs
(startup audit of 10-20 to 10-27 launch logs). Colocated runs do not take fix-022's concurrent start and keep the
old order.

## Decision
- With fix-022's concurrent start (fully async, separate GPUs, `--num-rollout`), the rollout side opens the engines
  on their loaded weights and starts rollout 0 before the trainer is ready (`publish_initial_weights`): it registers
  the zero LoRA adapter that every request names, publishes weight version 0 to the engines and the rollout
  executor, and marks the cells ready, which puts them in the router.
- When the trainer is ready the driver connects it and runs the usual version-0 sync, which sends the same weights
  as version 0, then resumes health monitoring that sync paused. The next sync is version 1, as before.
- The rollout executor waits for the trainer's parallel layout before it postprocesses a batch, since rollout 0 can
  finish before the trainer connects.
- Only where the engines provably hold the starting policy (`rollout_can_start_before_trainer`): a fresh run
  (`start_rollout_id` 0, no checkpoint tracker at the requested `--load`, no `--ckpt-step`), one policy, no eval
  before training, no debug modes, an unmerged adapter without `--lora-adapter-path`. Engines that have already
  served (a take-over) wait for the sync; that check takes the controller lock, so it runs before the weight-update
  window opens (the window detaches the lock, and the cells cannot be read without it). If the trainer restores or starts at any rollout but 0 (a surviving trainer
  can restore a later one while `start_rollout_id` stays 0), the driver stops.
- Lost: having the trainer build and send a zero adapter early (needs the trainer); full-weight runs too (the
  engines' FP8/MXFP4 load is not shown to equal the trainer's re-quantized version 0, and the step-0 log-probs would
  shift under the same version label).

## How it is applied
`train_async.py`, `miles/ray/placement_group.py`, `miles/ray/rollout/inference_controller.py`
(`awaits_first_weights`) and `miles/ray/rollout/rollout_executor.py`, every spot marked `[fork fix-063]`. Verify:
`tests/fast/test_train_async.py::TestRolloutAtEngineReady`, `tests/fast/ray/test_update_weights_ordering.py::TestRolloutAtEngineReady`,
`tests/fast/ray/rollout/test_inference_controller.py::TestAwaitsFirstWeights`, and in a split LoRA run "rollout starts on the engines' loaded weights" right after the engines are ready, the
step-0 train/rollout log-prob gap in its usual range and the next sync at version 1.

## Remove when
Upstream starts rollout before the trainer's first sync. Status 2026-10-28: not filed.
