import pytest


@pytest.fixture
def dsv41_lora_spec():
    """DeepSeek-V4.1 registered with native LoRA for one test; the registries are process-global."""
    from argparse import Namespace

    from miles.backends.megatron_utils.lora.native.registry import MODEL_SPECS
    from miles.utils.lora.arguments import load_custom_lora_spec
    from miles.utils.lora.hf_lora_targets import _HF_LORA_MODELS

    saved = [(registry, dict(registry)) for registry in (MODEL_SPECS, _HF_LORA_MODELS)]
    load_custom_lora_spec(Namespace(custom_lora_spec_path="miles_ext.lora.dsv41"))
    yield
    for registry, before in saved:
        registry.clear()
        registry.update(before)
