# Harness A/B: host steps 1-3 (CPU only), then step 4 (GPU, approved cap $60)

Owner: thread "Explore · host-side vs in-sandbox harness (cloud)". Plan: https://claude.ai/artifact/R7myy1fF1MZtYNSY54Sd5d
User approved 2026-10-19 10:27 AM PT: "Approved: harness A/B plan, GPU cap $60." CPU steps need no further approval.
Script: `harness-ab/harness_ab.py` on branch `harness-ab-probes` (`git show origin/harness-ab-probes:harness-ab/harness_ab.py > probes/harness_ab.py`). Code under test: listar2000/miles PR #51,
branch `claude/project-thread-ba70xa` (host harness parity). Run everything inside the trainer image built from that
branch (`MILES_FORK_COMMIT=<head of the branch>`), so harbor, miles_ext and mini-swe-agent 2.4.6 match production.
Names: Modal apps `mdev-harness-*`, Sail sandboxes `mdev-harness-*` (2 h lifetime, 15 min idle cap, smallest size).
Write results to `miles-fork-results/harness-ab/<step>/` and send the JSON summaries back to the cloud thread.

## Step 1: saved records (read-only, ~$1)
Copy (or mount) the Harbor trial dirs of a2, a3 and a4, then:
```
python probes/harness_ab.py timings <a2 trials dir>   > step1-a2.json
python probes/harness_ab.py timings <a3 trials dir>   > step1-a3.json
python probes/harness_ab.py timings <a4 trials dir>   > step1-a4.json
```
Also send, per run, the session server's own per-request latency (median and p90) by the same method the rollout
investigator used for "a2 agent waited 19.2 s, engine ~16.4 s". Agent-side `llm_wait_per_call_s` minus that is the
tunnel + proxy + session-server share. Also report how many trials failed in agent setup and the exception text.

## Step 2: network round trips (~$5 Modal CPU + Sail)
In one Modal CPU container (8 cores, trainer region), set the same HARBOR_* / Sail env as a run with the Sail provider
and the in-sandbox harness's tunnel (start `miles_ext.harbor.session_proxy` to a local echo server on the session port
range and `modal.forward` it, as `miles_ext/launch/modal_app.py` does). Echo server: any aiohttp route returning 200 on POST.
```
export PROBE_REPS=20 PROBE_ECHO_URL=<tunnel-url>/<token>/<port>/echo
for N in 1 32 128 256; do
  python probes/harness_ab.py trials --harness mini-swe-agent-host --agent-import-path harness_ab:ProbeAgent \
    --tasks tb4_ids.txt --tries $N --concurrency $N --model-host 127.0.0.1 --ports <first>-<last> --label probe-$N
done
```
(`tb4_ids.txt`: 32 TB4 instance ids. `harness_ab.py` must be importable: run from `probes/` or add it to PYTHONPATH.)
Each probe prints exec round-trip p50/p90/p99 for 0 B, 4 KB and 64 KB outputs and POST times for 50/300/800 KB.
Stop at the first concurrency that shows errors or rate limits and report them.

## Step 3: end-to-end replay, both harnesses (~$15 Modal CPU + Sail)
One Modal CPU container shaped like the head node (same CPU and memory as the B300 head's recipe shape).
1. Replay model on the session port range, from a2's trial dirs:
   `python probes/harness_ab.py replay-server --traj <a2 trials dir> --ports <first>-<last> &`
2. Session proxy + `modal.forward` pointing at 127.0.0.1 on that range, `MILES_SESSION_PUBLIC_URL=<tunnel>/<token>`.
3. Same task list (16 TB4 tasks a2 ran, 8 tries each = 128), both arms, one after the other:
```
for H in mini-swe-agent-sandbox mini-swe-agent-host; do
  python probes/harness_ab.py trials --harness $H --tasks tb4_16.txt --tries 128 --concurrency 128 \
    --model-host <container IP the proxy forwards to> --ports <first>-<last> --label $H-128 | tee step3-$H-128.jsonl
done
python probes/harness_ab.py trials --harness mini-swe-agent-host --tasks tb4_16.txt --tries 256 --concurrency 256 \
  --model-host ... --ports ... --label host-256 | tee step3-host-256.jsonl
```
4. Repeat the host 128 arm with a CPU burner holding the cores the optimizer uses on the head
   (`stress-ng --cpu <n>` or a busy-loop per core; n = head cores minus what a4's rollout process had).
Report the last JSON line of each run (try wall p50/p90, exit-status mix, CPU, RSS, loop lag) and the timings block.

## Step 4: GPU A/B (after steps 1-3; cap $60, max 3 launches)
Qwen3.5-9B, 1 x H100:4, rollout only (`--debug-rollout-only`, no trainer), TB4 tasks on Sail, 128 concurrent agents,
one ~1 h arm per harness (`taskset.harness`), same seed and tasks, top-k logprobs off. Register each run with the
monitor (`monitor/launch.sh`). Stop at once on a run-wide failure. The cloud thread sends the exact recipe before launch.
