"""Low-precision rollout: weight sync into a rollout engine that serves a quantized checkpoint.

checkpoint_quantizer  --custom-weight-quantizer-path for checkpoints whose layout no built-in Miles
                      quantizer covers (DeepSeek V4.1-Flash: FP8 dense + FP4 routed experts). The
                      rollout checkpoint's own tensor headers decide each tensor's format.
"""
