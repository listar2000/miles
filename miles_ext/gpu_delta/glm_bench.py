"""Run Miles' GPU-delta benchmark with GLM-5.3-Flash serving settings and FP8 mutations."""

import hashlib
import sys

import numpy as np

sys.path.insert(0, "/root/miles/tests/manual/gpu_delta")
import bench_gpu_delta as bench  # noqa: E402

bench.SERVER_ARGS = {
    "dtype": "bfloat16",
    "kv_cache_dtype": "bfloat16",
    "tp_size": 8,
    "dp_size": 8,
    "ep_size": 8,
    "pp_size": 1,
    "enable_dp_attention": True,
    "moe_runner_backend": "triton",
    "moe_a2a_backend": "none",
    "dsa_prefill_backend": "tilelang",
    "dsa_decode_backend": "tilelang",
    "mem_fraction_static": 0.7,
    "max_running_requests": 64,
    "chunked_prefill_size": 8192,
    "disable_cuda_graph": True,
    "disable_flashinfer_autotune": True,
    "tokenizer_worker_num": 1,
    "trust_remote_code": True,
    "skip_server_warmup": True,
    "reasoning_parser": "glm45",
}
bench.SERVER_ENV = {
    "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
    "SGLANG_HEALTH_CHECK_TIMEOUT": "900",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
    "PYTHONUNBUFFERED": "1",
}
_orig = bench._mutate


def _mutate(raw, name, dtype, seed, version, rate):
    if dtype != "F8_E4M3" or not name.endswith(".weight"):
        return _orig(raw, name, dtype, seed, version, rate)
    result = raw.copy()
    key = hashlib.sha256(f"{seed}:{version}:{name}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
    for start in range(0, result.size, bench.FRAME_BYTES):
        count = min(bench.FRAME_BYTES, result.size - start)
        positions = np.unique(rng.integers(0, count, size=max(1, round(rate * count)))) + start
        positions = positions[(result[positions] & 0x7F) != 0x7E]  # 0x7E ^ 1 would be NaN
        result[positions] ^= np.uint8(1)
    return result


bench._mutate = _mutate


async def _request(client, endpoint, payload=None, timeout=1200):
    http = client.http
    response = await (
        http.get(client.server_url + "/" + endpoint, timeout=timeout)
        if payload is None
        else http.post(client.server_url + "/" + endpoint, json=payload, timeout=timeout)
    )
    if response.is_error:
        raise RuntimeError(f"{endpoint} -> {response.status_code}: {response.text[:20000]}")
    return response.json() if response.content else None


bench._request = _request
if __name__ == "__main__":
    sys.argv = ["bench_gpu_delta.py", *sys.argv[1:]]
    bench.main()
