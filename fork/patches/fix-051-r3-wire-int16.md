# fix-051: R3 routing crosses the samples wire as int16 when the ids fit

## Why
Each `/sessions/{id}/samples` reply carries the trajectory's recorded expert routing (R3) as int32: tokens x
layers x top-k, about 175 MB at 170k tokens for DeepSeek V4.1 (43 layers x 6). In a 3-node V4.1 overfit run the
replies were 165-200 MB per sample. While the rollout loop was stalled, the session server reset a reply of
about 181 MB that the client was not reading, and the group lost that try.

## Decision
- `ValueSpec` gains `narrow`. A tensor field with `narrow` travels as that dtype when every value fits, and the
  decode widens it back to the field's dtype, so callers still see int32. Only `rollout_routed_experts` sets it
  (int16: V4.1 has 384 experts). Ids that do not fit travel as int32, as before.
- The wire gains the safetensors dtype `I16`. Upstream's codec TODO (removed here) planned the same dtype for #4035's int16 routed
  experts; whichever lands second reconciles.
- Cost: the decode copies the widened array instead of viewing the payload (#3974's zero-copy decode now
  holds for unnarrowed tensors only). The decode runs in a worker thread (`asyncio.to_thread` in
  `openai_endpoint_utils`), and numpy casts release the GIL.
- Not chosen: compressing the reply (CPU on both ends for routing ids that are close to random) and sending int16
  to the trainer (every R3 consumer reads int32).

## How it is applied
`miles/rollout/session/samples/codec.py`: `ValueSpec.narrow`, `_fits`, `_WIRE_DTYPES["I16"]`, and the narrowing
and widening steps in `encode_samples` and `decode_samples_and_merge_input_sample`, all marked `[fork fix-051]`.
Verify with `tests/fast/rollout/session/test_samples_codec.py::TestSamplesWireCodec::test_routed_experts_travel_as_int16_when_they_fit`.
On a sync, if upstream adds `I16` for #4035, keep one entry and re-check that the decode still widens.

## Remove when
Upstream sends routed experts in a narrower dtype or out of band. Upstream status: not filed.
