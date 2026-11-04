# ext-057: host memory breakdown of each trainer rank and node, once a minute

## Why
On GLM-5.3-Flash full-parameter runs, host memory decides how much of Adam's state stays in RAM (fix-073), and so the
step time. The recipe's 1,100 GiB reserve for everything besides Adam's host tier and GPU-delta's copies is a guess:
a full-parameter GLM-5.3-Flash split run was killed at its 2,304 GiB host-memory request, and nothing logged what held it.

## Decision
A daemon thread in each Megatron rank, plugged in with Miles' existing `--custom-megatron-init-path`. It samples every
5 s and logs once a minute the peak of:
- the rank's resident memory from `/proc/self/status` (anonymous, file, shared, locked);
- the pinned bytes PyTorch's host allocator holds;
- on local rank 0 only: the node's `/proc/meminfo` (page cache, dirty, shared), the cgroup v2 memory files when present,
  and the used bytes of `/dev/shm` and `/tmp` (Adam's disk tier).

Under Modal's gVisor, `/proc/self/status` has only `VmRSS` and meminfo's `Cached` does not show page cache; `AnonPages` and `Shmem` are the node fields that matter. The probe
(`fork/probes/hostram/hostram_probe.py`, CPU containers with an explicit limit `memory=(16, 24)` GiB) found:
- 48 GiB written and read back through buffered I/O survives the 24 GiB limit: buffered I/O does not get a container
  killed (its cache is either not counted or reclaimed under the limit; the probe does not tell which);
- 48 GiB in `/dev/shm`, or 32 GiB of anonymous memory, is killed at the limit;
- with a request and no limit, going 2-3x over the request was not killed, so a request alone is not enforced at
  its value. That says nothing about what counts;
- a B300:4 container accepts `memory=2560` GiB, but Modal refuses multi-node functions with fewer than 8 GPUs per node.

Logging peaks catches spikes between lines (a checkpoint save, a weight sync). It is read-only and never fails the
run. Rejected: W&B system metrics (one number per node, no page-cache or per-rank split).

## How it is applied
`miles_ext/obs/host_ram.py`; a recipe adds `extra_args: ["--custom-megatron-init-path", "miles_ext.obs.host_ram.install"]`.
Verify: the trainer log prints `[host ram] rank=R peak over 60s (GiB): ...` every minute per rank.
Test: `miles_ext/tests/test_host_ram.py`.

## Remove when
The host-RAM budget is measured and settled; upstream status: not filed (fork observability only).
