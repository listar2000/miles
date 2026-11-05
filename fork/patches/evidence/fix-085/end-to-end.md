# fix-085 evidence: end to end on one B300:8

Recipe `miles_ext/recipes/configs/glm53-flash-lora-dflash2.yaml` at this PR's first draft (`0938e4c130`, image built from
that commit): full GLM-5.3-Flash, 4 trainer GPUs (TP4 EP4), one TP4 DFlash2 engine, LoRA rank 32, MIS, score centering,
top-p 0.95, top-k 32, lr 1e-4, zero staleness, 5 rollouts of 4 STAK prompts x 8 samples, `reasoning_effort: low`.
W&B: https://forge.coreweave.com/wandb/mobiuslab/miles-glm53/runs/bfsdgs8t

"Gap" is `train/train_rollout_logprob_abs_diff`, and "max B" is the export log's largest `|lora_B|` for the adapter that
step trains from.

| Step | Gap | KL | Grad norm | max B | Raw reward | Replies at the 4k cap |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.026 | 0.0025 | 0.095 | 0 | 0.44 | 3% |
| 1 | 0.028 | 0.0029 | 0.053 | 1.0e-4 | 0.31 | 53% |
| 2 | 0.035 | 0.0042 | 0.059 | 2.0e-4 | 0.31 | 78% |
| 3 | 0.038 | 0.0052 | 0.071 | 3.0e-4 | 0.34 | 44% |
| 4 | 0.039 | 0.0056 | 0.066 | 4.0e-4 | 0.34 | 53% |

- Adapter: 970 tensors per sync, 0.4-0.9 s; `--check-lora-weight-equal` raised nothing.
- R3, from the saved rollout data: 0 of 69,927 rows all zero in the first rollout, 0 of 113,434 in the last.
- Decode: accept length 3.87-3.94 in the engine log.
- Resume: the same run id with `--num-rollout 6` loaded 970 adapter tensors and the optimizer from iteration 4 and ran
  step 5 at gap 0.025, starting from max B 5.0e-4.

Not established: why replies at the cap rise after the first rollout. It is one run, with 32 samples per rollout.
