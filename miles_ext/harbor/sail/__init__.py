"""Sail sandboxes for Harbor trials: ``miles_ext.harbor.sail:SailEnvironment``.

``harbor_sail.py`` is Harbor's own Sail provider (``src/harbor/environments/sail.py``
on Harbor main), vendored byte-identical: the Harbor branch this release pins
predates it. ``fork/vendored.json`` records its source commit and git blob, and
``fork/tools/ledger.py`` fails if the file drifts. Harbor loads environments it
does not know through ``EnvironmentConfig.import_path``, so no Harbor change is
needed; see fork/patches/ext-003-harbor-sail.md for the decision.

This subclass adapts the vendored class to the pinned Harbor and adds the
run's caps:

``type()``
    the pinned ``EnvironmentType`` has no ``SAIL`` member; Harbor's base class
    accepts any string from an import-path environment.
``max_lifetime_secs`` (environment kwarg)
    Sail terminates the Sailbox this long after creation, server-side, even
    if the trainer that created it is gone.
``idle_secs`` (environment kwarg)
    Sail may sleep a Sailbox only after this much idle time (compute billing
    stops while asleep; the next operation wakes it). Sail's own default is
    30 s, short enough to sleep a box whose agent is waiting on the model.

Credentials: the Sail SDK reads ``SAIL_API_KEY`` from the worker's environment.
"""

import asyncio
from typing import override

from sail import App, AutoSleep, ImageDefinition, Sailbox

from miles_ext.harbor.sail import harbor_sail as upstream

# the override calls the SDK through the vendored module's names (the same
# objects), so Harbor's unit tests, which patch harbor_sail.Sailbox, cover it

TYPE = "sail"
IMPORT_PATH = "miles_ext.harbor.sail:SailEnvironment"


class SailEnvironment(upstream.SailEnvironment):
    def __init__(self, *args, max_lifetime_secs: int | None = None, idle_secs: int | None = None, **kwargs):
        # popped before Harbor's __init__, which forwards the rest to BaseEnvironment
        self._caps = {}
        if max_lifetime_secs:
            self._caps["max_lifetime_seconds"] = int(max_lifetime_secs)
        if idle_secs:
            self._caps["auto_sleep"] = AutoSleep.not_before(int(idle_secs))
        super().__init__(*args, **kwargs)

    @staticmethod
    @override
    def type() -> str:
        return TYPE

    @override
    async def _create_sailbox(self, image: ImageDefinition) -> None:
        """Upstream's ``_create_sailbox`` plus ``max_lifetime_seconds`` and ``auto_sleep``.

        Re-check against harbor_sail.SailEnvironment._create_sailbox whenever
        the vendored file is updated: this override replaces it.
        """
        size, memory_limit_gib, disk_limit_gib = upstream._select_size(
            cpus=self._effective_cpus,
            memory_mb=self._effective_memory_mb,
            storage_mb=self._effective_storage_mb,
        )
        app = await upstream.App.find.aio(self._app_name, mint_if_missing=True)
        create_task = asyncio.create_task(
            upstream.Sailbox.create.aio(
                app=app,
                image=image,
                name=self.session_id,
                size=size,
                memory_limit_gib=memory_limit_gib,
                disk_limit_gib=disk_limit_gib,
                timeout=0,
                image_build_timeout=self._build_timeout_sec,
                egress_policy=self._to_egress_policy(self.network_policy),
                **self._caps,  # the only change from upstream
            )
        )
        try:
            self._sailbox = await asyncio.shield(create_task)
        except asyncio.CancelledError:
            try:
                self._sailbox = await asyncio.wait_for(create_task, timeout=30)
            except (asyncio.CancelledError, Exception):
                create_task.cancel()
            raise

