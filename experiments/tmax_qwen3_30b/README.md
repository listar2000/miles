# TMax SFT on Qwen3-30B-A3B-Thinking-2507

Supervised fine-tuning (SFT) of Qwen3-30B-A3B-Thinking-2507 on the passing trajectories of [allenai/tmax-sft](https://huggingface.co/datasets/allenai/tmax-sft) (ODC-BY).

| Terminal-Bench 2.1, 87 tasks × 2 trials | mean@1 | pass@2 |
| --- | --- | --- |
| Base | 1.7% | 3 |
| Original run with these settings (earlier trainer version), epoch 2 | 15.5% | 17 |

The eval used TMax's own agent (Vanillux2Agent, [hamishivi/tmax](https://github.com/hamishivi/tmax) @`fa86aa0`) with T 0.7, top-p 0.95, 16,384 tokens per call and 64 steps. That harness is not in this repository. This config reproduces the original run's data and settings, but has not been trained end to end on the current trainer.

## Run

Needs the model's weights on `MILES_WEIGHTS_VOLUME` at `/models/Qwen3-30B-A3B-Thinking-2507`, and `MILES_RESULTS_VOLUME`, `MILES_DOTENV_DIR` set.

```bash
python experiments/tmax_qwen3_30b/build_records.py --output records   # needs pyarrow, huggingface_hub
modal volume put miles-rl-datasets records /sft/tmax-qwen3-30b-recipe/records
export MILES_SFT_CONFIG=experiments/tmax_qwen3_30b/configs/qwen3-30b-tmax.yaml
python -m modal run -m miles_ext.sft.modal_prepare
python -m modal run --detach -m miles_ext.sft.modal_train --run-id <run-id>
```

Records: 5,611 training trajectories, 184 held out (40 tasks); 81,745 replies trained, 458 kept as untrained context ([build_records.py](build_records.py) says which). Prepared at 65,536 tokens: 5,599 samples.

Training: 2 epochs, 86 steps, batch 128, lr 2e-5 linear to 0, 3% warmup, on one H200:8 node (TP1 CP2 EP8). The final model is `hf/iter_0000085` in the run's results.
