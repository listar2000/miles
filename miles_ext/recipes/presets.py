"""Model presets: what a run needs to know about a model beyond its weights.

``tito_model`` is Miles' token-in/token-out template family; the two sglang
parsers are the ones Miles binds to that family
(miles/utils/chat_template_utils/tito_tokenizer.py). Miles v0.1.2 does not apply
them itself, so they are passed explicitly (RUN_LOG.md item 7).

``megatron_model_type`` names the model's Megatron definition,
``scripts/models/<megatron_model_type>.py`` (what Miles' ``load_model_args``
reads); a recipe that sets ``model.backend: megatron`` needs it.

The rest is what a model needs beyond the generic recipe, copied from the
model's upstream launcher (``scripts/run_<model>.py``), which stays the source
of truth: on a sync, diff these against it. ``hf_repo`` is what the rollout
engines serve; ``train_dir_name``, when set, is a separate BF16 copy under
/models that the trainer loads (the released checkpoint is quantized).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Preset:
    hf_repo: str
    tito_model: str
    tool_call_parser: str
    reasoning_parser: str
    backend: str = "fsdp"
    megatron_model_type: str | None = None
    chat_template_kwargs: dict = field(default_factory=dict)
    # cluster shapes this preset has been run on: (gpu type, nodes, backend)
    shapes: tuple[tuple[str, int, str], ...] = ()
    train_dir_name: str | None = None
    # rollout precision -> env the engines need for it; the first key is the default
    precisions: dict = field(default_factory=lambda: {"bf16": {}})
    # rollout precision -> smallest rollout.gpus_per_engine its weights fit in
    precision_min_gpus: dict = field(default_factory=dict)
    # rollout precisions whose checkpoint carries the speculative draft (DSpark/MTP) weights; empty: any
    draft_precisions: tuple[str, ...] = ()
    # --sglang-speculative-algorithm for the model's draft (recipe rollout.speculative); None: no draft
    speculative_algorithm: str | None = None
    # env for every process (engines and trainer), on top of the precision's
    env: dict = field(default_factory=dict)
    # env added only when weight syncs replace the engines' base weights (full-parameter or merged LoRA);
    # an adapter-served LoRA run (lora.serving adapter) leaves the base frozen and runs without it
    weight_sync_env: dict = field(default_factory=dict)
    # SGLang flags; {gpus} is the recipe's rollout.gpus_per_engine
    sglang_args: str = ""
    # Megatron flags the model's launcher adds; cp_args only when parallel.cp > 1
    megatron_args: str = ""
    megatron_cp_args: str = ""
    # False: the model's trainer takes one sample per micro-batch (--micro-batch-size 1)
    # instead of packing samples up to max_tokens_per_gpu (--use-dynamic-batch-size)
    packed_batches: bool = True
    attention_backend: str | None = "flash"  # --attention-backend; None: the model's spec decides
    # --custom-lora-spec-path for LoRA on a model Miles' native LoRA does not ship
    lora_spec_path: str | None = None

    @property
    def dir_name(self) -> str:
        return self.hf_repo.split("/")[-1]


PRESETS: dict[str, Preset] = {
    # Run 1 (overfit8-async-v1): FSDP, one H200:8 node, non-thinking
    "qwen3.5-9b": Preset(
        hf_repo="Qwen/Qwen3.5-9B",
        tito_model="qwen35",
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
        chat_template_kwargs={"enable_thinking": False},
        megatron_model_type="qwen3.5-9B",
        shapes=(("H200", 1, "fsdp"),),
    ),
    "qwen3-8b": Preset(
        hf_repo="Qwen/Qwen3-8B",
        tito_model="qwen3",
        tool_call_parser="qwen25",
        reasoning_parser="qwen3",
        chat_template_kwargs={"enable_thinking": False},
        megatron_model_type="qwen3-8B",
        shapes=(("H200", 1, "fsdp"),),
    ),
    # scripts/run_deepseek_v4_1.py (train) and docs/models/deepseek/deepseek-v4-1-flash.md.
    # The engines serve the released checkpoint (FP8 dense, FP4 experts); the trainer loads the
    # BF16 cast of it through mbridge.
    "deepseek-v4.1-flash": Preset(
        hf_repo="deepseek-ai/DeepSeek-V4.1-Flash-1017",
        train_dir_name="DeepSeek-V4.1-Flash-1017-bf16",  # the BF16 cast's directory name under /models
        tito_model="deepseekv41",
        tool_call_parser="deepseekv41",
        reasoning_parser="deepseek-v41",
        backend="megatron",
        megatron_model_type="deepseek-v4.1",
        lora_spec_path="miles_ext.lora.dsv41",
        chat_template_kwargs={"thinking": True, "reasoning_effort": "high"},
        # BF16 weights crash these two fusions in decode CUDA graph capture, adapter serving included
        precisions={
            "fp8_fp4": {"SGLANG_DSV4_FP4_EXPERTS": "1"},
            "bf16": {
                "SGLANG_DSV4_FP4_EXPERTS": "0",
                "SGLANG_OPT_FP8_WO_A_GEMM": "0",
                "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "0",
            },
        },
        # four B300 GPUs cannot hold the BF16 copy
        precision_min_gpus={"bf16": 8},
        # the BF16 copy has no MTP/DSpark tensors, and the release's FP8 draft does not boot on BF16 engines
        draft_precisions=("fp8_fp4",),
        speculative_algorithm="DSPARK",  # static verify; fix-053 keeps the R3 routing of verified tokens
        env={
            "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
            "NCCL_CUMEM_ENABLE": "1",
            "SGLANG_HEALTH_CHECK_TIMEOUT": "900",
            # hook-010 adapts wq_a and wkv one by one, and full weight updates cannot refresh the fused copy
            "SGLANG_OPT_FUSE_WQA_WKV": "0",
            "SGLANG_DISABLE_MULTIMEM_AG": "1",
            # engines here never sleep (not colocated), so the engram tables stay on the GPU
            "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "0",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "PYTHONFAULTHANDLER": "1",
            # the launcher's default --train-deterministic
            "NCCL_ALGO": "Ring",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
        # online weight updates cannot refresh weights these fusions derive (upstream launcher)
        weight_sync_env={"SGLANG_OPT_FP8_WO_A_GEMM": "0", "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "0"},
        # Text-only engines (vision_n_layers 0): with the vision tower loaded, every MoE layer routes through
        # SGLang's vision_topk, which never captures routed experts, so R3 gets all-zero routing; the loader
        # then skips the vision tensors. RL here is text-only and the trainer's BF16 cast has no vision tensors.
        sglang_args="""
            --sglang-ep-size {gpus} --sglang-dp-size 1
            --sglang-attention-backend dsv4 --sglang-moe-runner-backend auto
            --sglang-cuda-graph-max-bs-decode 128
            --router-health-success-threshold 1 --router-health-check-interval-secs 15
            --router-health-failure-threshold 40
            --sglang-json-model-override-args '{{"vision_n_layers": 0}}'
        """,
        # every V4.1 run so far: full-recompute layer inputs in host memory (ext-020) and a 60 min collective
        # timeout (the default 10 min trips on long checkpoint saves)
        megatron_args="""
            --dsv4-impl miles --model-name deepseekv41 --qkv-format bshd --transformer-impl transformer_engine --bf16
            --moe-router-freeze-gate --freeze-e-score-correction-bias
            --train-memory-margin-bytes 3221225472 --deterministic-mode
            --custom-megatron-init-path miles_ext.memory.ckpt_offload.install --distributed-timeout-minutes 60
        """,
        megatron_cp_args="--allgather-cp",
        packed_batches=False,
        attention_backend=None,
    ),
}
