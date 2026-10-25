"""Engine side of the split adapter sync (ext-037): replay each sync from the volume into the engines."""

import asyncio
import json
import threading
import time
from pathlib import Path

PUSH_TIMEOUT_SECS = 1800
MANIFEST = "manifest.json"
TENSORS = "adapter.safetensors"


class Agent:
    """The engine container's sync endpoint: /connect (the trainer is there) and /push (apply one sync)."""

    def __init__(self, engine_ports: list[int], *, gpus_per_engine: int, results_volume):
        self.engine_ports, self.gpus_per_engine = engine_ports, gpus_per_engine
        self.results_volume = results_volume
        self.lock = asyncio.Lock()

    def start(self, port: int) -> None:
        ready = threading.Event()
        threading.Thread(target=self._run, args=(port, ready), daemon=True).start()
        if not ready.wait(60):
            raise RuntimeError("split agent did not start")

    def _run(self, port: int, ready: threading.Event):
        from aiohttp import web

        async def connect(request):
            body = await request.json()
            if body.get("engines") != len(self.engine_ports):
                return web.json_response(
                    {
                        "ok": False,
                        "error": f"the container serves {len(self.engine_ports)} "
                        f"engines, the trainer expects {body.get('engines')}",
                    }
                )
            print(f"split agent: trainer connected ({len(self.engine_ports)} engines)", flush=True)
            return web.json_response({"ok": True})

        async def push(request):
            body = await request.json()
            async with self.lock:
                try:
                    out = await self.apply(Path(body["dir"]), selector=body.get("selector", "all"))
                except Exception as e:
                    return web.json_response({"ok": False, "error": f"{type(e).__name__}: {e}"[:2000]})
            return web.json_response({"ok": True, **out})

        async def main():
            app = web.Application()
            app.router.add_post("/connect", connect)
            app.router.add_post("/push", push)
            runner = web.AppRunner(app, access_log=None)
            await runner.setup()
            await web.TCPSite(runner, "127.0.0.1", port).start()
            ready.set()
            await asyncio.Event().wait()

        asyncio.run(main())

    async def apply(self, sync_dir: Path, *, selector: str) -> dict:
        from safetensors.torch import load_file

        t0 = time.monotonic()
        await asyncio.to_thread(self.results_volume.reload)
        manifest = json.loads((sync_dir / MANIFEST).read_text())
        tensors = await asyncio.to_thread(load_file, str(sync_dir / TENSORS))
        t1 = time.monotonic()
        for names in manifest["buckets"]:
            bucket = [(n, tensors[n]) for n in names]
            await asyncio.gather(*(self._send(port, i, bucket, selector) for i, port in enumerate(self.engine_ports)))
        t2 = time.monotonic()
        print(
            f"split agent: v{manifest['version']} applied, reload+read {t1 - t0:.1f}s, send {t2 - t1:.1f}s", flush=True
        )
        return {"version": manifest["version"], "read_s": t1 - t0, "send_s": t2 - t1}

    async def _send(self, port: int, engine_index: int, bucket: list, selector: str):
        """Send one bucket to one engine: the full tensors to each TP rank on its own GPU, over CUDA IPC."""
        import httpx
        import torch
        from sglang.srt.utils import MultiprocessingSerializer

        try:
            from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
        except ImportError:
            from sglang.srt.model_executor.model_runner import FlattenedTensorBucket

        keep, serialized = [], []
        for rank in range(self.gpus_per_engine):
            device = torch.device("cuda", engine_index * self.gpus_per_engine + rank)
            flat = FlattenedTensorBucket(named_tensors=[(n, t.to(device)) for n, t in bucket])
            data = {"flattened_tensor": flat.get_flattened_tensor(), "metadata": flat.get_metadata()}
            keep.append(data)
            serialized.append(MultiprocessingSerializer.serialize(data, output_str=True))
        payload = {
            "serialized_named_tensors": serialized,
            "load_format": "flattened_bucket",
            "flush_cache": False,
            "selector": selector,
        }
        async with httpx.AsyncClient(timeout=PUSH_TIMEOUT_SECS) as client:
            resp = await client.post(f"http://127.0.0.1:{port}/update_weights_from_tensor", json=payload)
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, dict) and result.get("success") is False:
            raise RuntimeError(f"engine {engine_index}: {result.get('message') or result}")
        del keep
        torch.cuda.synchronize()
