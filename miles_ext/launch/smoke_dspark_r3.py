"""Bench-only patch for SGLang 7b6993a: DSpark decode results drop the verify pass's capture outputs.

The target verify forward fills routed_experts_output (R3 rows, pending D2H in overlap mode), indexer_topk_output
and kv_weight_version_record, but DSpark's TargetVerifyResult keeps only logits, and _forward_decode builds its
GenerationBatchResult without them, so every decoded token's R3 row stays zero (EAGLE/DFlash pass them through:
eagle_worker_common.py, dflash_worker_v2.py). Idempotent; run before the engines start.
"""
import sys
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang/python/sglang/srt/speculative/dspark_components")
EDITS = {
    "dspark_verify.py": [
        ("""class TargetVerifyResult(msgspec.Struct, frozen=True):
    logits_output: object
    can_run_cuda_graph: bool
""", """class TargetVerifyResult(msgspec.Struct, frozen=True):
    logits_output: object
    can_run_cuda_graph: bool
    routed_experts_output: object = None
    indexer_topk_output: object = None
    kv_weight_version_record: object = None
"""),
        ("""        return TargetVerifyResult(
            logits_output=target_out.logits_output,
            can_run_cuda_graph=target_out.can_run_cuda_graph,
        )""", """        return TargetVerifyResult(
            logits_output=target_out.logits_output,
            can_run_cuda_graph=target_out.can_run_cuda_graph,
            routed_experts_output=target_out.routed_experts_output,
            indexer_topk_output=target_out.indexer_topk_output,
            kv_weight_version_record=target_out.kv_weight_version_record,
        )"""),
    ],
    "dspark_worker_v2.py": [
        ("""            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=accept.new_seq_lens,
        )""", """            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=accept.new_seq_lens,
            routed_experts_output=target_verify.routed_experts_output,
            indexer_topk_output=target_verify.indexer_topk_output,
            kv_weight_version_record=target_verify.kv_weight_version_record,
        )"""),
    ],
}
for name, edits in EDITS.items():
    p = D / name
    s = p.read_text()
    for old, new in edits:
        if new in s:
            continue
        assert s.count(old) == 1, f"{name}: anchor not found once: {old[:60]!r}"
        s = s.replace(old, new)
    p.write_text(s)
print("dspark_r3 patch applied to", D)
