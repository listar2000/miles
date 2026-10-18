# hook-003: `--custom-pre-trainer-hook-path`, rollout 0 starts before the trainer exists

## Why
In a split run (`--deploy-component primary|inference|trainer`, each its own Ray cluster) the
driver connects to the trainer first, pushes weights, then starts rollout 0. A trainer that runs in
its own Modal app is paid from its first second, so it sits idle for the whole first rollout
(minutes with 8k replies). On a fresh run the engines already serve `--hf-checkpoint`, the very
weights the trainer would push, so rollout 0 does not need the trainer.

## Decision
A function-path flag in the style of Miles' other `--custom-*-path` flags. `train_async.py` awaits
it with `(args, inference_controller, rollout_executor)` right after the rollout components exist.
It may return the rollout-0 task it started: the driver then skips the initial weight push and uses
that task as the first batch. Returning `None` (or leaving the flag unset) keeps the default order.
First user, ext-021: `miles_ext.disagg.late_start.start_rollout_before_trainer`, which publishes
weight version 0 and the trainer's data split, starts rollout 0 and waits for the trainer's address.
Rejected: a new deploy mode in train_async (a larger upstream diff), and starting the trainer early
and idling it (what the hook removes).

## How it is applied
`miles/utils/arguments.py` (flag) and `train_async.py` (`load_function` import, the call, the
guarded initial `update_weights`, the first-batch task); all spots carry `[fork hook-003]`. The hook
must return `None` on resume (a checkpoint exists), since the engines then serve stale weights.
Verify: a split run with the hook logs `late-start hook: rollout 0 started` before the trainer's
address. On a sync, re-insert the call if upstream reorders train_async's start-up.

## Remove when
Upstream starts rollout 0 without the trainer in split mode, or offers an equivalent extension
point. Status: not filed.
