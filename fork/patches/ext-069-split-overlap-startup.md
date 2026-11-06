# ext-069: the split trainer starts with its engines, not after them

## Why
A split launch waited for the engines to serve before it started the trainer, and the trainer's relay waited again
before `ray start`. The engines then idled through Ray, the router, the session servers and the model build.

One-cluster runs have no such gap: the worker manager starts engines, routers and session servers in one step. On the
GLM-5.3-Flash DFlash2 LoRA run of 2026-11-05 ([W&B project](https://forge.coreweave.com/wandb/mobiuslab/miles-glm53)),
the engine and router commands both ran at 20:35:54.

One uncontrolled pair of GLM-5.3-Flash LoRA runs on 4 + 4 B300, read from their Modal logs. The new run also carries
the part-node commits it needs for this shape, at batch 4 x 8 instead of 16 x 8, and other image commits differ. The
batch size matters only after rollout 0. No repeat, so no spread; treat the numbers as one observation:

| Measure | Old order ([run](https://forge.coreweave.com/wandb/mobiuslab/miles-glm53/runs/zx4q4lgo)) | New order |
| --- | --- | --- |
| engine container running to engines ready | 8m19 | 6m31 |
| Ray ready, relative to engines ready | 3m34 after | 4m38 before |
| rollout 0, after engines ready | 8m55 | 32 s |
| trainer's first sync, after engines ready | 13m10 | 3m17 |
| engine container running to first sync | 21m29 | 9m48 |

## Decision
- The launcher starts the trainer once the engine container runs (`relay.wait_container`). Waiting for the container
  keeps a trainer from being billed while the engines queue for GPUs. The platform's `spawn_detached` already spawned
  both at once, so it gains the overlap too.
- The relay connects on a background thread. Its local ports and the agent URL come from the recipe, so Ray starts
  right away, and Miles retries the closed relay ports (`_fetch_server_info_with_retry`) until the engines serve.
  fix-086 raises that retry budget from 10 min to 1 h, since a cold DeepSeek-V4.1 engine start reaches 16 min.
- A relay that fails stops the run with an "error" record naming the cause: rank 0 calls `Supervisor.fail`, and another
  node reports through the progress dict (`relay.report_node_failure`), which rank 0's heartbeat reads. Reports are
  keyed by a token rank 0 publishes before it opens the head's port, so a late report from an earlier segment is never
  read.
- No switch: it applies to every split recipe (`fork/README.md`, flags).
- Trade-off: with `t_E` the engine load and `t_T` the trainer's start to its first sync, the engines save
  `min(t_E, t_T)` of billed time and the trainer pays `max(0, t_E - t_T)` idle. The launch costs more when
  `trainer_rate * max(0, t_E - t_T)` exceeds `engine_rate * min(t_E, t_T)`, which equal allocations can also reach: a
  16 min engine load and a 4 min trainer save 4 engine-minutes and add 12 trainer-minutes.
- Lost: a relay that opens at once and holds requests until the engines serve (more code, and Miles' per-request
  timeouts would retry anyway); a recipe key for the old order (nothing needs it).

## How it is applied
`miles_ext/launch/modal_app.py` (`main`, `_split_links`), `miles_ext/launch/split/relay.py` (`wait_container`,
`connect_in_background`, `agent_url`), `miles_ext/launch/split/engine.py` (`container_at`),
`miles_ext/platform/supervisor.py` (`fail`), `miles/ray/rollout/external_engine_provider.py` (fix-086). Tests:
`miles_ext/tests/test_split_relay.py`, `miles_ext/tests/test_supervisor.py`. In a split launch's log, "Ray cluster ready" and the
router come before "engines ready", and "rollout 0 starts" follows the engines closely.

## Remove when
With ext-036.
