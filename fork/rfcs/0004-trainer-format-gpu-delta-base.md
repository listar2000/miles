# RFC 0004: Engines serve the trainer's own FP8 bytes, so the first GPU-delta sync is empty

Status: accepted 2026-11-03 (owner). Step 1 done by hand for GLM-5.3-Flash; steps 2-3 are approved, not yet built.

## Problem

GPU-delta (port-001, port-002, ext-053) sends the XOR of the bytes the engines serve and the bytes the trainer
exports. The engines start from the released FP8 checkpoint; the trainer loads that checkpoint, holds BF16
weights and re-quantizes them to FP8 on export. The roundtrip is lossy (the release was quantized from
higher-precision masters, and dequantized values round to BF16), so the block scales and most FP8 codes move.

Measured on GLM-5.3-Flash before any training step, in the first full-model split runs (2026-11-03): about 95%
of the served model's bytes differ, a 240 GB first delta (30 GB of LZ4 from each of the 8 trainer ranks that
own a share of the weights). The receiver cannot pin arenas
that large under Modal, and even where it can, the first sync would cost about what a full sync costs.
Later step deltas are small: in the same runs, step 1 changed ~36 MB and sent ~0.5 GB per trainer rank.

## Options

| Option | Result | Why not / cost |
| --- | --- | --- |
| A. Accept the large first sync | 240 GB through the volume and receiver once per run | Fails on Modal (host arenas); a cold start pays a full sync |
| B. Make the trainer reproduce the release bytes | First delta empty | Impossible in general: the release came from higher-precision weights |
| C. Engines serve the trainer's own FP8 export (chosen) | First delta empty; every later delta is a training delta | A second checkpoint per model (328 GB for GLM), built once |

C relies on the trainer's export being idempotent: loading its own FP8 output and exporting again gives the
same bytes. Checked on GLM-5.3-Flash: the re-export of the trainer-format base changed 0 of 40,131,840,656
bytes on every trainer rank, in both runs that started from it. With FP32 amax/448 scales this holds because a BF16 rounding error (2^-9
relative) is far below FP8's code spacing, and amax maps back to the same scale.

## Design

1. **Build (done by hand, 2026-11-03).** The base was built on CPU from a published first delta: for every
   tensor of the publication, release bytes XOR the decoded LZ4 frames, checked against the manifest's
   `changed_bytes`. The one-off script is kept with the campaign notes (listar2000/miles-ops
   `notes/glm53/trainer_base_from_publication.py`). Output: `miles-rl-weights:zai-org--GLM-5.3-Flash/29523a33f41b2cea7f85f9288433a50fb506eb39`
   (a synthetic revision id; `.ready` names the release and the publication).
2. **Launcher step (approved, to build).** A one-off CPU function `prepare_trainer_base(preset, revision)`:
   run the trainer's own exporter on CPU (no publication, no engines) and write safetensors beside the
   release with the same index, under a revision derived from the release revision and the exporter's
   source hash. A recipe opts in with `model.base: trainer` instead of a hand-written revision.
3. **DeepSeek V4.1 (approved, after 2).** The same step applies once V4.1 has a GPU-delta mapping; its MXFP4 experts
   need their own idempotence check.

## Tradeoffs and open questions

- Engines serve re-quantized weights, not the release. Not yet measured: eval quality of the trainer-format
  base against the release (expected equal within noise; one eval before relying on it).
- Storage: one extra copy per model and exporter version; old copies are removed only with the owner's OK.
- If the exporter changes (quantizer, scale format), the base must be rebuilt; deriving the revision from the
  exporter's hash makes a stale base visible instead of silently large.
