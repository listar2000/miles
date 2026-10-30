# SFT on pre-rendered agent trajectories (exploratory)

A replication of a prime-rl SFT recipe (Qwen3-30B-A3B-Thinking-2507 on a strict-thinking subset of
nvidia/Nemotron-SFT-SWE-v3, evaluated on SWE-Bench Verified tasks 0-99 with mini-swe-agent 2.4.6),
run on Modal from the published main image with this directory overlaid. Not wired into the recipe
system or the platform launcher yet.

| File | Role |
| --- | --- |
| `render.py` | Row normalization, and `TrajectoryRenderer`: replays the session server's turn-by-turn TITO tokenization (`TITOTokenizer.merge_tokens`) on a stored message list, so SFT tokens and loss masks follow the RL/serving layout |
| `prepare.py` | Modal CPU build: weights download, a map over the 96 source shards (validate, decontaminate, render, drop > 64K), and a reduce (one row per task and harness, content dedup, task-level 2% validation split, digest re-check, `manifest.json`) |
| `rollout.py` | `--rollout-function-path`: Samples from the memory-mapped token shards, nothing re-rendered |
| `ids.py`, `val.py` | Validation pseudo-rollouts: ids >= `VAL_BASE` run forward-only in `SftTrainRayActor` and log `val/nll` at `train/step` |
| `train.py` | The driver: Miles' `train.py` loop with `--debug-train-only`, validation before step 1 and every `MILES_SFT_VAL_INTERVAL` steps |
| `modal_train.py` | The H200:8 Modal function and the full Miles argv (TP1/CP2/EP8, fused log-probs, cosine LR) |
| `parity.py` | Real eval rollouts vs the renderer, plus trial and timing statistics |
| `common.py` | Names, revisions, the image and the overlay |
| `configs/` | The eval recipes (fork launcher, `--debug-rollout-only`, Sail sandboxes) |

Outside this directory: the preset `qwen3-30b-a3b-thinking-2507` (`miles_ext/recipes/presets.py`) and
`MILES_OVERLAY_EXT=1` in `miles_ext/launch/modal_app.py`, which copies this checkout's `miles_ext`
over the image so an unpushed branch runs.

```bash
python -m modal run -m miles_ext.sft.prepare::weights
python -m modal run -m miles_ext.sft.prepare::build
python -m modal run --detach -m miles_ext.sft.modal_train --run-id <id>      # same id again resumes
MILES_OVERLAY_EXT=1 MILES_WEIGHTS_VOLUME=miles-rl-weights MILES_IMAGE=<main image digest> \
MILES_DATA=<dir> MILES_RECIPE=miles_ext/sft/configs/swebv100-qwen3-sft-0000999.yaml \
  python -m modal run --detach -m miles_ext.launch.modal_app --run-id <eval id>
python -m modal run -m miles_ext.sft.parity --run-id <eval id>               # parity gate
```

Eval tasks: Harbor `swebench-verified@1.0`, tasks 0-99 in Hugging Face order, with each Dockerfile's
FROM changed to `ghcr.io/epoch-research/swe-bench.eval.x86_64.<instance_id>:latest` (the same
SWE-bench-harness images; Sail's anonymous Docker Hub pulls are rate-limited). Prompt rows are
`{"prompt": <instruction.md>, "metadata": {"instance_id": <task dir>}}`.

Known gaps: the data has no tool schemas (mini-swe-agent sends one at eval); a parsed message rebuilds
`text\n\n<tool_call>` as `text\n<tool_call>`, so SFT teaches one newline where the model writes two;
steps are sized in rows (12), not token positions.
