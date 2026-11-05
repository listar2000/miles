"""GLM-5.3-Flash engine probe on one B300:8 (no training): plain decode vs native MTP (EAGLE) vs DFlash2.

Arms, in order, so the gate comes first and a late failure only loses speed numbers:
  full    one DFlash2 TP4 engine beside one plain TP4 engine, both with LoRA on (no adapter loaded), the FP32 LM head
          score centering forces, and top-p replay. Calibration, R3, prefill agreement, speed per engine.
  dflash  two DFlash2 TP4 engines: speed.
  mtp     two TP4 engines with the release's NextN layer (EAGLE, 3 steps, 4 draft tokens): speed.
  plain   two plain TP4 engines: speed.

Calibration: under exact sampling E[p(sampled)] = sum p^2, so the mean difference over generated tokens is 0.

  MILES_FORK_COMMIT=<sha> python -m modal run --detach dflash2_probe.py --run-id dflash2-probe-1105-a1
"""

import json
import math
import os
import random
import statistics
import subprocess
import time

import modal

from miles_ext.platform.images import fork_image

MODEL = "/models/zai-org--GLM-5.3-Flash/46a050bec13002ad29ee795c16e61cf4fa3c4a9c"
DRAFT = "/models/incoai--GLM-5.3-Flash-DFlash2/bf582e4eacc1810f76656d1811693ff6c6737d2a"
PROMPTS = "/root/stak32.jsonl"
TARGETS = "/root/lora_targets.json"
HARD_LIMIT_S = 60 * 60  # no new work past this, about the $80 cap with the function timeout below

app = modal.App("miles-rl-glm53-dflash2-probe")
image = (
    fork_image("d8030f5b5cb7723dc29bcce7e3b2beefd7c8ad9f")
    .add_local_file(os.path.expanduser("~/data/glm53-lora/stak32.jsonl"), PROMPTS)
    .add_local_file(os.path.expanduser("~/data/glm53-lora/lora_targets.json"), TARGETS)
)
weights = modal.Volume.from_name("miles-rl-weights")
results = modal.Volume.from_name("miles-rl-results")

COMMON = [
    "--trust-remote-code", "--mem-fraction-static", "0.75", "--chunked-prefill-size", "8192",
    "--dsa-prefill-backend", "tilelang", "--dsa-decode-backend", "tilelang", "--kv-cache-dtype", "bfloat16",
    "--moe-runner-backend", "triton", "--enable-return-routed-experts", "--context-length", "32768",
    "--max-running-requests", "128",
]
DFLASH = ["--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", DRAFT,
          "--speculative-draft-attention-backend", "fa4"]
MTP = ["--speculative-algorithm", "EAGLE", "--speculative-num-steps", "3", "--speculative-eagle-topk", "1",
       "--speculative-num-draft-tokens", "4"]


def _full_extra() -> list[str]:
    return ["--enable-fp32-lm-head", "--enable-lora", "--max-lora-rank", "16", "--max-loras-per-batch", "1",
            "--lora-target-modules", *json.load(open(TARGETS))]


def _launch(extras: list[list[str]], log_dir: str) -> list[tuple[subprocess.Popen, int]]:
    env = {**os.environ, "SGLANG_HEALTH_CHECK_TIMEOUT": "900", "TORCHINDUCTOR_COMPILE_THREADS": "1",
           "PYTHONFAULTHANDLER": "1"}
    procs = []
    for i, extra in enumerate(extras):
        port = 31000 + i
        cmd = ["python", "-m", "sglang.launch_server", "--model-path", MODEL, "--tp", "4", "--ep-size", "4",
               "--base-gpu-id", str(i * 4), "--port", str(port), "--host", "127.0.0.1", *COMMON, *extra]
        log = open(f"{log_dir}/engine{i}.log", "w")
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), port))
    return procs


def _wait(procs, timeout=1500) -> float:
    import requests

    t0, pending = time.time(), {p for _, p in procs}
    while pending:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"engines not healthy: {pending}")
        for proc, port in procs:
            if proc.poll() is not None:
                raise RuntimeError(f"engine on {port} exited {proc.returncode}")
        for port in list(pending):
            try:
                if requests.get(f"http://127.0.0.1:{port}/health_generate", timeout=60).status_code == 200:
                    pending.discard(port)
            except Exception:
                pass
        time.sleep(10)
    return time.time() - t0


def _r3_zero(meta, n_in, n_out):
    import base64

    import numpy as np

    info = meta.get("routed_experts")
    if info is None:
        return n_out
    rows = np.frombuffer(base64.b64decode(info), dtype=np.int32).reshape(n_in + n_out - 1, -1)[n_in:]
    return int((rows == 0).all(axis=1).sum())


async def _load(ports, texts, concurrency, duration, max_new, params, body_extra):
    import asyncio

    import aiohttp

    stats = {"tokens": 0, "tok_s": [], "lat": [], "r3_tokens": 0, "r3_zero": 0, "errors": [], "verify": 0, "spec": 0}
    stop_at = time.time() + duration

    async def worker(k):
        port, i = ports[k % len(ports)], k
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200)) as s:
            while time.time() < stop_at:
                body = {"text": texts[i % len(texts)], "return_routed_experts": True,
                        "sampling_params": {"temperature": 1.0, "max_new_tokens": max_new, **params}, **body_extra}
                i += concurrency
                t0 = time.time()
                try:
                    async with s.post(f"http://127.0.0.1:{port}/generate", json=body) as r:
                        out = await r.json()
                    meta = out["meta_info"]
                    n_out, n_in = meta["completion_tokens"], meta["prompt_tokens"]
                except Exception as e:
                    stats["errors"].append(repr(e)[:300] + " " + str(locals().get("out"))[:300])
                    await asyncio.sleep(1)
                    continue
                dt = time.time() - t0
                stats["tokens"] += n_out
                stats["lat"].append(dt)
                stats["tok_s"].append(n_out / dt if dt else 0)
                stats["r3_tokens"] += n_out
                stats["r3_zero"] += _r3_zero(meta, n_in, n_out) if n_out else 0
                if meta.get("spec_verify_ct"):
                    stats["verify"] += meta["spec_verify_ct"]
                    stats["spec"] += n_out

    t0 = time.time()
    await asyncio.gather(*[worker(k) for k in range(concurrency)])
    wall = time.time() - t0
    return {
        "concurrency": concurrency, "requests": len(stats["lat"]), "errors": len(stats["errors"]),
        "first_errors": stats["errors"][:2], "tok_s": round(stats["tokens"] / wall, 1),
        "per_request_tok_s_p50": round(statistics.median(stats["tok_s"]), 1) if stats["tok_s"] else None,
        "latency_p50_s": round(statistics.median(stats["lat"]), 1) if stats["lat"] else None,
        "r3_zero_row_share": round(stats["r3_zero"] / max(1, stats["r3_tokens"]), 4),
        "tokens_per_verify": round(stats["spec"] / stats["verify"], 2) if stats["verify"] else None,
    }


async def _collect(port, texts, n, max_new, params, body_extra):
    import asyncio

    import aiohttp

    sem = asyncio.Semaphore(32)

    async def one(k, s):
        body = {"text": texts[k % len(texts)], "return_logprob": True, "return_routed_experts": True,
                "sampling_params": {"temperature": 1.0, "max_new_tokens": max_new, "ignore_eos": True, **params},
                **body_extra}
        async with sem, s.post(f"http://127.0.0.1:{port}/generate", json=body) as r:
            return {**await r.json(), "_text": body["text"]}

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200)) as s:
        return await asyncio.gather(*[one(k, s) for k in range(n)])


def _calibration(outs, mode):
    diffs, outside, sum_err, zero, tokens, errors = [], 0, 0.0, 0, 0, []
    for out in outs:
        meta = out.get("meta_info")
        if not meta or "output_token_logprobs" not in meta:
            errors.append(str(out)[:300])
            continue
        selected = meta["output_token_logprobs"]
        n_out, n_in = meta["completion_tokens"], meta["prompt_tokens"]
        tokens += n_out
        zero += _r3_zero(meta, n_in, n_out)
        for t, entry in enumerate(selected):
            lp, token = entry[0], entry[1]
            if mode == "top":
                row = [e[0] for e in meta["output_top_logprobs"][t]]
                p = math.exp(lp)
            else:
                ids, row = meta["output_token_sampling_mask"][t], meta["output_token_sampling_logprobs"][t]
                if token not in ids:
                    outside += 1
                    continue
                p = math.exp(row[ids.index(token)])
                sum_err = max(sum_err, abs(math.log(sum(math.exp(x) for x in row))))
            diffs.append(p - sum(math.exp(2 * x) for x in row))
    if not diffs:
        return {"mode": mode, "errors": errors[:2], "samples": len(outs)}
    mean, se = statistics.fmean(diffs), statistics.stdev(diffs) / math.sqrt(len(diffs))
    return {"mode": mode, "tokens": len(diffs), "mean": round(mean, 5), "se": round(se, 5), "z": round(mean / se, 2),
            "outside_support": outside, "max_abs_log_support_sum": sum_err, "r3_zero_row_share": round(zero / max(1, tokens), 4),
            "errors": len(errors)}


async def _prefill_gap(port, outs, n, tok):
    """Mean |selected logprob - prefill logprob| on the plain engine, which has the same FP32 LM head."""
    import aiohttp

    gaps = []
    picked = random.Random(0).sample(outs, min(n, len(outs)))
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        for out in picked:
            selected = (out.get("meta_info") or {}).get("output_token_logprobs") or []
            if not selected:
                continue
            ids = tok(out["_text"], add_special_tokens=False).input_ids + [e[1] for e in selected]
            body = {"input_ids": ids, "return_logprob": True, "logprob_start_len": 0,
                    "sampling_params": {"temperature": 1.0, "max_new_tokens": 1}}
            async with s.post(f"http://127.0.0.1:{port}/generate", json=body) as r:
                res = await r.json()
            full = (res.get("meta_info") or {}).get("input_token_logprobs") or []
            tail = full[-len(selected):]
            gaps += [abs(a[0] - b[0]) for a, b in zip(selected, tail) if b[0] is not None and a[1] == b[1]]
    return {"tokens": len(gaps), "mean_abs_gap": round(statistics.fmean(gaps), 4) if gaps else None}


def _run_full(texts, procs, report_arm, tok):
    import asyncio

    dflash_port, plain_port = procs[0][1], procs[1][1]
    support_params = {"top_p": 0.95, "top_k": 32}
    support_body = {"return_sampling_mask": True, "sampling_logprobs_mode": "support", "return_logprob": True}
    # top-p replay with R3, as the training recipe would ask, one engine at a time
    for name, port in (("dflash", dflash_port), ("plain", plain_port)):
        asyncio.run(_load([port], texts, 8, 45, 256, support_params, support_body))
        report_arm[f"speed_{name}_topp"] = [asyncio.run(_load([port], texts, c, 150, 2048, support_params, support_body))
                                            for c in (32, 64)]
    # calibration, as in fix-058: 64 samples x 512 tokens on each engine, score-centering mode and top-p mode
    for name, port in (("dflash", dflash_port), ("plain", plain_port)):
        top = asyncio.run(_collect(port, texts, 64, 512, {}, {"top_logprobs_num": 64}))
        report_arm[f"calibration_{name}_top64"] = _calibration(top, "top")
        sup = asyncio.run(_collect(port, texts, 64, 512, {"top_p": 0.95, "top_k": 64}, support_body))
        report_arm[f"calibration_{name}_support"] = _calibration(sup, "support")
        report_arm[f"prefill_gap_{name}"] = asyncio.run(_prefill_gap(plain_port, top, 16, tok))


def _run_speed(texts, procs, report_arm, tok):
    import asyncio

    ports = [p for _, p in procs]
    asyncio.run(_load(ports, texts, 8, 45, 256, {}, {}))
    report_arm["speed"] = [asyncio.run(_load(ports, texts, c, 180, 2048, {}, {})) for c in (64, 128)]


@app.function(image=image, gpu="B300:8", volumes={"/models": weights, "/results": results},
              timeout=70 * 60, cpu=64, memory=1024 * 1024)
def probe(run_id: str, arms: str):
    from transformers import AutoTokenizer

    start = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    rows = [json.loads(line) for line in open(PROMPTS)]
    texts = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True, reasoning_effort="low")
             for r in rows]
    out_dir = f"/results/probes/{run_id}"
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    for name in arms.split(","):
        if time.time() - start > HARD_LIMIT_S:
            report[name] = {"skipped": "hard limit"}
            continue
        extras = {
            "full": lambda: [[*DFLASH, *_full_extra()], _full_extra()],
            "dflash": lambda: [DFLASH, DFLASH],
            "mtp": lambda: [MTP, MTP],
            "plain": lambda: [[], []],
        }[name]()
        log_dir = f"{out_dir}/{name}"
        os.makedirs(log_dir, exist_ok=True)
        procs = _launch(extras, log_dir)
        report[name] = {}
        try:
            report[name]["ready_s"] = round(_wait(procs))
            (_run_full if name == "full" else _run_speed)(texts, procs, report[name], tok)
        except Exception as e:  # an arm that fails is a result too
            report[name]["error"] = repr(e)[:2000]
        finally:
            for proc, _ in procs:
                proc.kill()
            for proc, _ in procs:
                proc.wait()
            subprocess.run(["pkill", "-9", "-f", "sglang"], check=False)
            time.sleep(15)
        print(json.dumps({name: report[name]}), flush=True)
        with open(f"{out_dir}/report.json", "w") as f:
            json.dump(report, f, indent=1)
        results.commit()
    return report


@app.local_entrypoint()
def main(run_id: str, arms: str = "full,dflash,mtp,plain"):
    print(json.dumps(probe.remote(run_id, arms), indent=1))
