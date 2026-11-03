# GPU delta end-to-end benchmark

Status: **PASS**. Every arm's final selected text, token IDs and full input/output logprobs are compared with one independently loaded target checkpoint on every route; see `comparison.json`.

One row per arm/version, including the first update. SHA on/off controls both sender payload hashing and receiver verification; manifest SHA remains enabled. Versions are cumulative synthetic targets, not learned updates or statistical repeats. Selected outputs do not prove every weight byte. Startup and oracle generation are outside the update timings.

Prepared inputs: `/tmp/bench/inputs`; reused: False. Input construction is shared across all arms and excluded from compression measurements; its metadata remains in `inputs.json`.

|Codec|Frame KiB|SHA|Version|Sender timing origin|Compression s|Prepare s|Pause ms|Inner GiB|Outer GiB|Payload GiB|
|---|---|---|---|---|---|---|---|---|---|---|
|snappy-zstd|1024.0|off|1|current_run|5.199|1.888|3684.844|1.212|0.263|0.264|
|snappy-zstd|1024.0|off|2|current_run|0.721|0.251|68.383|1.212|0.263|0.264|
|snappy-zstd|1024.0|off|3|current_run|0.691|0.254|39.687|1.212|0.263|0.264|

In saved-delta apply mode, every sender/read/encode/seal/compression timing is historical from the referenced fixture, never current-run encoder performance. Validation I/O is recorded separately in fixture-verification.json; no encoding time is invented or included in current totals. New receiver apply/resume/generation measurements remain current. The final oracle is opt-in.

Compression is the maximum owner compression/publication wall time, including optional payload SHA, H2D/XOR/final D2H, writes and shard flush/fsync. It excludes prepared-input reads/pinning, shared input construction and parent manifest sealing. Prepare is the maximum receiver background-preparation wall time. Pause is the maximum actual scheduler pause from completed pause/resume timestamps. These independent owner/rank maxima can come from different processes and must not be added.

Sizes sum across all owners; GiB = 2^30 bytes. Inner is the compressed frame bytes before outer Zstd. Outer is the stored Zstd matrix payload; it is absent for plain LZ4. Payload includes all stored matrix/raw bytes and alignment, excluding the manifest. The packed inner arena, exact byte counts, manifest/publication sizes, frame sizes and canonical/changed bytes remain in `summary.json`.

`summary.json` retains every owner/rank receipt at full precision, including input read/pin and owner elapsed time, publication writes, finalization, pack/D2H, parent sealing, CPU outer decode/plain copy and coordinator time. Components overlap. CUDA intervals from older fixtures/receipts remain historical diagnostics, including wrapper/launch gaps; fresh runs leave them unmeasured and use Nsight for GPU profiling. The table rounds values to three decimal places.

|Codec|Version|Kernel prepare ms|Kernel cache misses/rank|Paused setup ms|Apply wall ms|Layout CUDA ms|GPU pipeline ms|Descriptor KiB|Apply metadata KiB|
|---|---|---|---|---|---|---|---|---|---|
|snappy-zstd|1|1360.728|3.000|7.632|3680.655|unmeasured|unmeasured|5.031|10.859|
|snappy-zstd|2|0.024|0.000|9.182|50.235|unmeasured|unmeasured|5.031|10.859|
|snappy-zstd|3|0.019|0.000|10.988|36.463|unmeasured|unmeasured|5.031|10.859|

Kernel preparation occurs before the serving pause and includes the cached kernel lookup or first compilation/loading. Cache misses count per-process module warmup requests per update; a request may compile or load a disk-cached binary. Missing counters in older receipts remain unmeasured. Descriptor bytes are included in total apply metadata bytes. Each cell is an independent rank maximum; nested host spans and overlapping CUDA events must not be added.

