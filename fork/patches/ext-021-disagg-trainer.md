# ext-021: disaggregated trainer on Modal (`miles_ext/disagg`)

## Why
An always-on RL job keeps the trainer GPUs paid while the engines generate, which in fully async
math RL with long replies is most of every step. Miles' split deployment mode
(`--deploy-component`) already separates the trainer from the rollout side; this runs that mode on
Modal so the trainer can live in its own app, join late, or run only for each training step.

## Decision
- One Modal app holds the rollout side: the SGLang engines (inference deployment) and the driver
  (primary deployment), each on its own Ray cluster. The trainer is a second app that registers its
  address in a Modal Dict. A CPU conductor (a pure state machine plus a loop) starts the trainer
  when batch 0 is near, retries it and stops every app by id at the end.
- Weight sync is NCCL over Modal's private IPv6 network (i6pn), which links containers of
  different apps only within one cloud and region: cross-app runs pin `cloud="aws",
  region="us-west"` (CPU probes: 8/8 cross-app clients at 5-7 Gbit/s; oci us 8/8 but long H100
  queues; aws us-east 2/4; gcp us-east 0/8). `DISAGG_CLOUD`, `DISAGG_REGION` and `DISAGG_PIN=0`
  override it.
- Rollout batches travel through Mooncake: the master and a segment host (16 GB) run in the rollout
  container; the trainer contributes no segment.
- On-demand trainer: per step it starts, loads model and Adam from a Volume, trains, saves, pushes
  weights and exits; the driver saves before pushing, so a trainer lost mid-step is retried from
  the last save and no update is applied twice. The driver sets the run's weight version
  (step + 1) after each push, since every new trainer counts its pushes from 1.
- A trainer lost during the NCCL push leaves every engine in the push stuck (SGLang's weight-sync
  group takes no timeout), so a medic next to the engines restarts them on the driver's request
  and also starts engine cells Miles tore down; the step is then retried.
- Rejected: disk-delta weight sync for the on-demand trainer (it needs a trainer that stays up),
  and one Ray cluster spanning apps (Ray over i6pn works, but split mode keeps the apps decoupled).

## How it is applied
New files under `miles_ext/disagg/` only, plus hook-003, hook-004 and fix-013 in upstream files.
Tests: `miles_ext/tests/test_disagg_conductor.py` (conductor with a fake trainer) and
`test_disagg_ondemand.py` (lost trainer, superseded attempts, engine restart, medic). Design facts,
launch commands and measured numbers: `miles_ext/disagg/README.md`.

## Remove when
Never as a whole (it is our launcher); drop parts that upstream covers.
