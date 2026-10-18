# miles_ext/disagg: the Miles trainer in its own Modal app

Runs Miles' split deployment (`--deploy-component primary | inference | trainer`, one Ray cluster
each, `--worker-comm-backend rpc`, shared `--run-uuid`) on Modal, so the trainer GPUs need not be
paid while the engines generate. Fork id ext-021; patch page `fork/patches/ext-021-disagg-trainer.md`.

Terms:
- **primary**: the driver process that runs the training loop (`train_async.py`, or `ondemand.py`).
- **inference**: the SGLang engines. **trainer**: the FSDP trainer workers.
- **conductor**: a 1-CPU Modal function that starts the other functions, retries the trainer and
  stops every app by id at the end.
- **i6pn**: Modal's private IPv6 network between containers; it links apps only within one cloud
  and region.

## Modes

| mode | what it tests | Modal apps / functions | GPUs (bench) |
| --- | --- | --- | --- |
| always-on | the baseline: one container, one Ray cluster, `train_async.py` | `miles-disagg` (or `$DISAGG_APP`) / `baseline` | H100 x (engines + trainer) = 4 |
| one-container split | split deployment without cross-app networking | `miles-disagg` / `single` | H100 x 2 |
| late start (`--phase late`) | trainer app joins a rollout that is already generating | `miles-disagg-<run>`: `conductor` (CPU), `rollout` (engines + primary, Mooncake); `miles-disagg-<run>-trainer` (deployed): `trainer` | rollout H100 x 2, trainer H100 x 2 |
| per-step trainer (`--phase per-step`) | a fresh trainer per training step | `miles-disagg-<run>`: `conductor_per_step`, `rollout` (primary runs `ondemand.py`); `miles-disagg-<run>-trainer`: `step_trainer` | rollout H100 x 2, each step's trainer H100 x 2 |

GPU counts per app come from `MILES_DISAGG_SHAPE` (JSON `{"rollout_gpus": n, "trainer_gpus": m}`),
read when Modal defines the functions. Cross-app apps pin `cloud="aws", region="us-west"`.

## Data paths

- **Rollout batches**: Mooncake. The master (`--rpc_address ::`, `MC_USE_IPV6=1`) and a segment
  host (`segment_host.py`, 16 GB) run in the rollout container; the trainer contributes no segment.
- **Weights**: NCCL broadcast from trainer rank 0 to the engines over i6pn (`NCCL_SOCKET_FAMILY=AF_INET6`;
  fix-013 brackets the IPv6 rendezvous address). About 0.5 s per push for 4B in always-on and late start,
  about 17 s per step trainer (a new NCCL group each time).
- **Checkpoints** (per-step trainer only): volume `miles-disagg-state`, `/state/<run>/ckpt` (model + Adam, ~47 GB
  per step for 4B); the step trainer keeps the newest 2. Models and data: volume `miles-disagg-models`.
  Replay recordings: `/state/replay[/<tag>]/rollout_<k>.pt`.
- **Coordination**: Modal Dict `miles-disagg-<run>`:

| key | written by | meaning |
| --- | --- | --- |
| `rollout` | rollout | i6pn host and Mooncake address |
| `trainer_addr` | late-start trainer | its controller address (the late-start hook waits for it) |
| `want/<k>` | per-step driver | step k needs a trainer, attempt n |
| `trainer_addr/<k>/<n>` | step trainer | its controller address |
| `commit/<k>`, `committed/<k>` | driver / step trainer | step k saved; volume committed |
| `release/<k>` | driver | step k done; the trainer exits |
| `lost/<k>/<n>` | conductor | that trainer's function ended before release |
| `restart_engines/<tag>`, `engines_restarted/<tag>` | driver / medic | restart every engine after a cut-off push |
| `steps/<k>`, `paid/<k>/<n>`, `events/*`, `calls/*` | all | timings, paid seconds, events, spawned call ids (a restarted conductor reattaches) |
| `stop` | rollout | the run is over |

## One per-step trainer step

1. The driver waits for batch k (rollout keeps generating while no trainer exists), writes `want/k`.
2. The conductor spawns `step_trainer(k, n)`; a trainer whose attempt is already superseded exits.
3. The trainer reloads the volume, starts Ray and its deployment, publishes `trainer_addr/k/n`.
4. The driver takes it over: from step 1 it loads model, Adam and RNG from `/state/<run>/ckpt`;
   it asserts the restored rollout id is k (or k+1, see 7).
5. Train, save, `commit/k`; the trainer prunes old iterations, commits the volume, `committed/k`.
6. Push weights, set the run's weight version to k+1 on engines and rollout executor, `release/k`.
7. Trainer lost (function ended, or no address within 20 min): the conductor marks `lost/k/n` and
   cancels superseded calls; the driver retries with attempt n+1, which restores k (retrain) or
   k+1 (it had saved: push only). If the push itself was cut off, the driver first has the medic
   restart every engine and waits for them.

Upstream touch points: hook-003, hook-004, fix-013 (`python fork/tools/ledger.py markers`; pages in `fork/patches/`).

## Modules

| module | role |
| --- | --- |
| `app.py` | image, volumes, region pin; `check` (argv parse), `single`, `baseline` |
| `argv.py` | per-component Miles argv; presets `bench` and `replay` |
| `split.py` | late-start and per-step apps: `rollout`, `trainer`, `step_trainer`, `conductor`, `conductor_per_step` |
| `conductor.py` | late-start conductor state machine (pure) |
| `late_start.py` | pre-trainer hook: admit engines, start rollout 0, wait for the trainer |
| `ondemand.py` | per-step driver |
| `medic.py` | restarts engine cells |
| `procs.py` | Ray clusters, Mooncake master, driver processes, stop app by id |
| `segment_host.py`, `run_exit.py`, `addrs.py` | Mooncake segment, hard exit after a driver, pool address lookup |
| `gate.py`, `models.py` | CPU test gate in the image, model download |

## Launch

```
cd <worktree>; export MILES_DISAGG_SHAPE='{"rollout_gpus": 2, "trainer_gpus": 2}'
modal run -m miles_ext.disagg.app --mode check                                   # CPU argv check
DISAGG_APP=miles-disagg-b modal run --detach -m miles_ext.disagg.app --mode baseline --bench --run-id bench-b-1
MILES_DISAGG_RUN=bench-late-1 modal run --detach -m miles_ext.disagg.split --phase late --bench --expected-batch-s 120
MILES_DISAGG_RUN=bench-per-step-1 modal run --detach -m miles_ext.disagg.split --phase per-step --bench [--num-rollout 4]
# replay: record once (always-on), then train any mode on the same batches
DISAGG_APP=miles-disagg-rec modal run --detach -m miles_ext.disagg.app --mode baseline --replay record:r2 --run-id rec-1
MILES_DISAGG_RUN=replay-per-step-1 modal run --detach -m miles_ext.disagg.split --phase per-step --replay replay:r2 --num-rollout 3 --kill 1:committed
modal run -m miles_ext.disagg.gate --tests "miles_ext/tests/test_disagg_ondemand.py"  # unit tests in the image (CPU)
```

## Measured (Qwen3-4B, dapo-math-17k, H100, aws/us-west, 2026-10-18)

Benchmark (32 prompts x 8, 8k replies, staleness 2, fully async, 10 steps):

| | always-on | late start |
| --- | --- | --- |
| train per step | ~155 s | ~155 s |
| step period | 206 s | 222 s |
| weight sync | 0.48 s | 0.53 s |
| train-vs-rollout logprob gap | 0.013 | 0.013 |
| reward | 0.43-0.70 | 0.43-0.68 |
| stale groups dropped | 8 | 47 (36 at step 3, after the trainer joined) |
| trainer start | with the run | rollout 0 at 2:11:18, trainer address 2:20:28, step 0 logged 2:23:57 |

Replay (the same 5 recorded batches, 8 prompts x 8, 8k replies), grad norm vs the recording:

| step | always-on | late start | per-step | per-step, push killed at step 1 |
| --- | --- | --- | --- | --- |
| 1 | 0.01% | 0.001% | 0.004% | 0.0014% |
| 2 | 1.05% | 0.3% | 2.0% (retried after a kill) | 0.16% |
| 3 | 0.55% | 2.4% | 0.24% | |
| 4 | 2.2% | 0.7% | | |

(train/loss is ~0 by construction here: ratio 1 at the first micro step and group-normalized advantages.)

Per-step trainer, per step (container start to release, queue excluded): start to ready ~185 s,
load 20-64 s, train 56-97 s (155 s at bench shape), save 81-133 s, commit 4-9 s, push ~17 s.
Fixed overhead O ~ 350 s.

| | value |
| --- | --- |
| trainer GPU time per step, bench shape | per-step ~505 s vs always-on 206 s (2.5x more) |
| break-even (equal trainer GPU time) | rollout:train ~3.3x (bench: ~1.3x) |
| 40% fewer trainer GPU-hours | rollout:train >= ~5.4x |
| GPU wait per trainer request, 10-18 night | 0-47 min (one step never started) |
| engine restart after a cut-off push | 19 s to ack, then the next push |

## Known limits

- Cross-app runs need one cloud and region with i6pn between apps (aws/us-west; aws/us-east 2/4).
- The per-step trainer costs more than always-on unless rollout per step is >~3.3x the train step,
  and every step waits for a free GPU; under scarcity the engine GPUs idle meanwhile.
- No exact trainer-vs-engine weight check: `--check-weight-update-equal` scrambles engines before
  rollout 0, which late start serves. Evidence used: logprob gap and replay grad norms.
- SGLang's weight-sync group takes no timeout. A late-start trainer lost mid-push leaves the
  engines hung until the process-group timeout (~20-30 min); only the per-step driver asks the medic
  for a restart.
- A restart reloads every engine (fleet-wide), not just the stuck ones.
- FSDP only (the late-start hook knows the FSDP data split).
