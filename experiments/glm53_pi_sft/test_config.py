from pathlib import Path

from miles_ext.sft import config as sft_config


def test_every_run_config_loads_and_fits_its_cap():
    configs = sorted((Path(__file__).parent / "configs").glob("*.yaml"))
    assert configs
    for path in configs:
        cfg = sft_config.load(path)
        assert cfg.max_tokens_per_gpu * cfg.train.cp >= cfg.data.max_tokens
