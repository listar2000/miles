# fix-079 evidence

Revision under test: `fix-079-gpu-delta-device-arena.patch` sha256 `a2d41f1cc658a068…` (this PR's head), with port-001, fix-074 and
hook-018, applied by `apply.py` to SGLang `3a938213c1` in `radixark/miles:v0.1.3@sha256:bb235de60c2d…` plus nvCOMP
6.0.0.12. Miles side: the #145 branch at `eb82ce949e` (port-002, fix-075 to fix-077, the GLM benchmark wrapper).

- `unit-tests.txt`: SGLang GPU-delta host, memory and receiver tests with this revision (38 passed), and the new
  fallback test failing on the first revision.
- `cuda-result.txt`: how `CUDA_ERROR_OUT_OF_MEMORY` prints and converts in our image.
- `bench-host/`, `bench-staged/`: Miles' GPU-delta benchmark on the FP8 GLM-5.3-Flash 4-layer slice, one EP8 engine
  on B300:8, three synthetic versions, final-target check on (`miles_ext/gpu_delta/glm_bench.py` from #145; the staged
  run adds `GPU_DELTA_DEVICE_ARENA=1` to its server environment):

      python3 miles_ext/gpu_delta/glm_bench.py --model <4-layer FP8 slice> --output /tmp/bench/run \
        --prepared-inputs /tmp/bench/inputs --codecs snappy-zstd --hashing off --versions 3 --ports 31000 \
        --validate-final-target

  `bench-host` leaves `GPU_DELTA_DEVICE_ARENA` unset: the 320 MB rank arena was admitted as a host mapping, so
  nothing was staged (`host_rank_device_staged` 0 on all 24 rank receipts). `bench-staged` sets it to 1: every rank
  copied its ~320 MB used extent into a 335,544,320-byte device image (`staging.json`, from the full `summary.json`).
  Both PASS (`comparison.json`). Pause per `REPORT.md` (maximum over ranks, one run, version 1 includes kernel
  loading): host 61.8 / 74.6 ms, staged 68.4 / 39.7 ms on versions 2 / 3.

The automatic fallback (variable unset, mapping refused) is covered by the unit test only: at this slice's size
Modal admits the mapping, and the full model's refusal was seen before this revision.
