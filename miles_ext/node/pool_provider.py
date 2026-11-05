"""Registered rollout nodes as Miles' cells, on upstream's engine provider hook (ext-066)."""

import asyncio
import os

from miles_ext.launch.split.volume_sync import REGISTRY_ENV
from miles_ext.node import contract as C
from miles_ext.node import registry as node_registry
from miles_ext.recipes.recipe import SPLIT_ENGINE_PORT

from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.polling_reconcile_loop import PollingReconcileLoop
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, CellReconcileFn, StopWatchFn
from miles.utils.workers.worker_spec import HostAndPort

_POOL_ID = "registry-inference-engine-0"
_MODEL_ID = "default"
POLL_SECS = 5.0


def registry_engine_provider(args, *, capability) -> "RegistryEngineProvider":
    return RegistryEngineProvider(args)


def cells_of(members: list[dict], *, api_key: str | None) -> dict[str, CellInfo]:
    cells = {}
    for m in members:
        gpus = m["gpus_per_engine"]
        for slot, port in zip(m["slots"], m["engine_ports"], strict=True):
            index = slot - SPLIT_ENGINE_PORT
            cell_id = compute_cell_id(pool_id=_POOL_ID, cell_index=index)
            cells[cell_id] = CellInfo(
                cell_id=cell_id,
                pool_id=_POOL_ID,
                alive=True,
                worker_names=[compute_worker_name(pool_id=_POOL_ID, cell_index=index)],
                workers_hash=f"{m['node_id']}:{port}",
                meta=dict(
                    model_id=_MODEL_ID,
                    worker_type="regular",
                    num_gpus_per_engine=gpus,
                    gpu_offset=index * gpus,
                    sglang_api_key=api_key,
                    needs_offload=False,
                    update_weights=True,
                    slot=slot,
                ),
            )
    return cells


class RegistryEngineProvider(BaseWorkerProvider):
    def __init__(self, args) -> None:
        self._api_key = args.sglang_api_key
        self._url, self._token = os.environ[REGISTRY_ENV], os.environ[C.POOL_TOKEN_ENV]
        self._cells: dict[str, CellInfo] = {}

    async def _refresh(self) -> dict[str, CellInfo]:
        members = await asyncio.to_thread(node_registry.members, self._url, self._token)
        self._cells = cells_of(members, api_key=self._api_key)
        return self._cells

    async def init(self) -> None:
        await self._refresh()

    def _addrs(self, cell_id: str):
        return dict(primary=HostAndPort(host="127.0.0.1", port=self._cells[cell_id].meta["slot"]))

    async def get_addrs(self, worker_name: str):
        for cell_id, info in self._cells.items():
            if worker_name in info.worker_names:
                return self._addrs(cell_id)
        raise AssertionError(f"{worker_name} is not an engine of a registered rollout node")

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return [
            [
                WorkerInfo(name=name, generation=0, self_addrs=self._addrs(cell_id), gpu_ids=[])
                for name in self._cells[cell_id].worker_names
            ]
            for cell_id in cell_ids
        ]

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        return await PollingReconcileLoop(list_cells=self._refresh, poll_interval_seconds=POLL_SECS).start(reconcile)

    def expected_num_cells(self, *, group_id: str) -> int:
        return sum(1 for info in self._cells.values() if info.meta["model_id"] == group_id)
