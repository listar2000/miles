from pathlib import Path

from miles_ext.sft import config as sft_config

CONFIGS = sorted((Path(__file__).parent / "configs").glob("*.yaml"))


def test_the_run_configs_load():
    assert [p.name for p in CONFIGS] == ["qwen3-30b-128k.yaml"]
    cfg = sft_config.load(CONFIGS[0])
    assert (cfg.model, cfg.train.cp, cfg.max_tokens_per_gpu) == ("qwen3-30b-a3b-thinking-2507", 4, 32768)
