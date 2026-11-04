# SFT of Qwen3-30B-A3B-Thinking-2507 on GLM-5.3-Flash Pi traces

Started 2026-11-03. Distil GLM-5.3-Flash's passing Terminal-Bench-style trajectories (Pi harness, Mobius
tasksets tb-2-v3-2 to v3-8) into Qwen3-30B-A3B-Thinking-2507 with Miles SFT.

## Basics

| | |
| --- | --- |
| Student | `Qwen/Qwen3-30B-A3B-Thinking-2507` (preset `qwen3-30b-a3b-thinking-2507`, TITO `qwennext`, 256K native context), weights on `miles-rl-weights:/Qwen3-30B-A3B-Thinking-2507` |
| Teacher | `zai/glm-5.3-flash` in Pi 1.5.0, Mobius evals, 2 trials per task, rewards 0/1 |
| Data | 7 evals (below): 4,326 training records, 179 held out; raw ATIF on the devbox `~/data/sft-glm53-pi/raw/`, records on `miles-rl-datasets:/sft/tb2-pi-glm53flash/records` |
| Training | `miles_ext.sft` on one Modal H200:8 node, config [configs/qwen3-30b-128k.yaml](configs/qwen3-30b-128k.yaml) |

| Eval id | Trials | Train records | Held out |
| --- | --- | --- | --- |
| 8ddd1c6c | 1000 | 653 | 25 |
| eab74b3b | 999 | 669 | 32 |
| f9a24f19 | 1000 | 660 | 26 |
| 2ae82da6 | 1000 | 662 | 23 |
| d7c175cf | 1000 | 637 | 24 |
| 952ef66c | 1000 | 650 | 35 |
| 9f038ca9 | 574 | 395 | 14 |

## Pipeline

Data lives outside the repo, in `~/data/sft-glm53-pi/` on the devbox: `raw/`, `records/`.

1. **Raw export** ([export_raw.sh](export_raw.sh) `<eval-uuid>...`): every trial of an eval (all outcomes) in Harbor layout,
   `mobius eval traces <id> --download-dir`, once per eval id (`.done` marker). Trials with trace status
   `missing` are runtime errors; Mobius then reports `complete: false`, which is expected.
2. **Records** (`PYTHONPATH=<miles checkout> python experiments/glm53_pi_sft/build_records.py [--root ~/data/sft-glm53-pi]`):
   filters, then `miles_ext.sft.connectors.atif`, one
   `records/<eval_id>.jsonl` + `.val.jsonl` + reports per eval and a line in `records/_manifest.jsonl`. New
   evals are picked up by running it again. Then `modal volume put` the records folder.
3. **Shards**: `MILES_SFT_CONFIG=experiments/glm53_pi_sft/configs/qwen3-30b-128k.yaml python -m modal run -m miles_ext.sft.modal_prepare`.
4. **Train**: the same config with `python -m modal run --detach -m miles_ext.sft.modal_train --run-id <id>`.

## Filters and decisions

| | Rule | Dropped (7 evals) |
| --- | --- | --- |
| F1 | outcome `passed` and reward 1.0 (also drops reward-1.0 trials where Pi exited non-zero) | 1,905 trials |
| F3 | malformed tool call: arguments not an object, unknown tool. Missing `required` keys are not checked: Pi's strict schemas mark nullable keys required (`read`'s `offset`/`limit`) and Pi accepts them omitted | 1 |
| F10 | last agent step is a tool call: the run hit the agent timeout (21 of 23 in v3-2 stop at 43-60 min; median run 15 min) | 148 |
| F9 | task prompt shares >= 20% of its word 8-grams with a TB 2.1 instruction (our eval benchmark) | 0 |
| F2 | connector refusals (image tool results, `multimodal`) | 14 |
| F8 | held out: tasks with sha1(task) in the first 3%, all their trials | 179 records |

Decided by the owner (2026-11-03):
- **TB canary kept.** It reaches trajectories through task files (479 of 703 v3-2 passing trials) and the
  model copies it into files it writes; it matters for RL, not for this SFT.
- **Steps without reasoning are trained** (32% of steps). They are mostly obvious follow-ups (build after an
  edit, read after a listing); code edits keep their reasoning (13-16% skip it). The student learns when not
  to think.
- **Every passing trial is kept** (no per-task dedup).
- **Pi's strict tool schemas kept**, as GLM saw them (optional arguments required and nullable, null arguments
  in calls kept). Miles' Pi harness sends plain schemas by default; evaluate this model with its strict-mode
  option on.
- **128K cap.** Qwen tokenizes these traces 9-14% longer than GLM; at 128K about 88% of trajectories fit
  (54% at 64K). Samples over the cap are dropped whole.

## What the data looks like (v3-2, 703 passing trials)

- Pi tools: `bash` 52% of calls, `read` 15%, `edit` 13%, `write` 13%, `ls`, `grep`, `find`. Median 24 agent steps.
- Final context (GLM tokens): median 54K, max 314K.
- Reasoning: median 800 characters when present, p90 about 10K.
- The last turn is a markdown report of the work; about 8% of tool results are errors the model recovers from.
- The system prompt names Pi's own docs and tools; some tasks use the network.
- Rendered for Qwen: `<|im_start|>assistant\n<think>\n` is prompt, the trained part is
  `reasoning\n</think>\n\n<tool_call>...</tool_call><|im_end|>`; a no-reasoning turn trains
  `\n</think>\n\n<tool_call>...`. Each trajectory is one sample (thinking retention `all`, as Qwen3-2507 is
  served with `clear_thinking=False`).

## Training

The tmax Qwen3-30B run that worked (TB 2.1 1.7% to 15.5% mean@1): lr 2e-5 linear to 0, warmup 3%, 2 epochs,
batch 128 trajectories. Here the same at 128K: TP1, CP4, EP8 (DP2), 32K tokens per GPU (as the 64K tmax runs
at CP2), full recompute. Prepared at 128K: 3,911 training samples (415 too long), 222.9M tokens of which 159.7M
trained, and 168 held-out samples. 3,911 / 128 = 30 steps per epoch, 60 steps; held-out NLL every 10 steps.

Throughput (2-step smoke): ~300 s/step, 23K tokens/s, ~360 TFLOPs per GPU (about 36% of the 989 TFLOPs peak
Miles uses). Selective recompute (MoE activations and norms) ran out of memory at 32K tokens per GPU; other
layouts were not measured (no H200 capacity at the time).

## Evaluation (open)

Evaluate in Pi, the harness that made the traces: serve the HF export behind an OpenAI-compatible endpoint and
point a Mobius eval at it. Baseline: Qwen3-30B-A3B-Thinking-2507 in the same setup.

## Runs

| Run | Config | Result |
| --- | --- | --- |
| `smoke-sft-glm53pi-q3-128k` | qwen3-30b-128k, 2 steps | loss 1.156 -> 1.151, held-out NLL 1.153 -> 1.030 |
| `sft-glm53pi-q3-128k-ep2` | qwen3-30b-128k | launched 2026-11-04 01:25 UTC |
