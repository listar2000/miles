"""Processes a deployment container runs: one Ray cluster per deployment, the Mooncake master, Miles drivers."""

import json
import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field

MILES_SRC = "/root/miles"
MOONCAKE_PORT = 50051
# per-cluster port block: Ray head, raylet, object manager, client, agents, worker range
_BLOCK = 3000
_MILES_PORT_BLOCK = 4000


def i6pn_addr() -> str:
    return socket.getaddrinfo("i6pn.modal.local", None, socket.AF_INET6)[0][4][0]


def wait_port(host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            socket.create_connection((host, port), timeout=3).close()
            return
        except OSError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"{host}:{port} not reachable after {timeout:.0f}s") from None
            time.sleep(1)


@dataclass
class RayCluster:
    """A single-node Ray cluster that owns ``gpus`` (physical ids) on this host."""

    name: str
    index: int
    node_ip: str
    gpus: tuple[int, ...] = ()
    cpus: int = 8

    @property
    def port(self) -> int:
        return 6379 + self.index

    @property
    def address(self) -> str:
        host = f"[{self.node_ip}]" if ":" in self.node_ip else self.node_ip
        return f"{host}:{self.port}"

    def env(self) -> dict:
        env = {
            **os.environ,
            "RAY_ADDRESS": self.address,
            "MILES_HOST_IP": self.node_ip,
            "PYTHONPATH": f"{MILES_SRC}:/root/Megatron-LM",
            # hook-004: Miles rpc/engine ports, disjoint per deployment sharing this host
            "MILES_DYNAMIC_PORT_START": str(20000 + self.index * _MILES_PORT_BLOCK),
            "MC_USE_IPV6": "1",  # Mooncake's IPv6 mode (i6pn has no IPv4)
        }
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpus))
        return env

    def start(self) -> None:
        base = 10000 + self.index * _BLOCK
        cmd = [
            "ray",
            "start",
            "--head",
            f"--port={self.port}",
            f"--node-ip-address={self.node_ip}",
            f"--num-gpus={len(self.gpus)}",
            f"--num-cpus={self.cpus}",
            "--disable-usage-stats",
            "--include-dashboard=true",
            "--dashboard-host=127.0.0.1",
            f"--dashboard-port={base + 7}",
            f"--temp-dir=/tmp/ray-{self.name}",
            f"--node-manager-port={base}",
            f"--object-manager-port={base + 1}",
            f"--ray-client-server-port={base + 2}",
            f"--dashboard-agent-listen-port={base + 3}",
            f"--dashboard-agent-grpc-port={base + 4}",
            f"--metrics-export-port={base + 5}",
            f"--runtime-env-agent-port={base + 6}",
            f"--min-worker-port={base + 100}",
            f"--max-worker-port={base + _BLOCK - 1}",
        ]
        subprocess.run(cmd, env=self.env(), check=True)

    def stop(self) -> None:
        subprocess.run(["ray", "stop", "--force"], env=self.env(), check=False)

    def pool_addr(self, pool_id: str, *, driver: "Driver", timeout: float = 900) -> tuple[str, int]:
        """The rpc (host, port) of ``pool_id``'s first worker, once the driver's worker manager has placed it."""
        deadline = time.monotonic() + timeout
        while True:
            out = (
                subprocess.run(
                    ["python", "-m", "miles_ext.disagg.addrs", pool_id],
                    env=self.env(),
                    cwd=MILES_SRC,
                    capture_output=True,
                    text=True,
                )
                .stdout.strip()
                .splitlines()
            )
            if out:
                addr = json.loads(out[-1])
                return addr["host"], addr["port"]
            if driver.proc.poll() is not None:
                raise RuntimeError(f"{driver.name} exited ({driver.proc.returncode}) before {pool_id} was placed")
            if time.monotonic() > deadline:
                raise TimeoutError(f"{pool_id} not placed after {timeout:.0f}s")
            time.sleep(5)


def start_mooncake_master(port: int = MOONCAKE_PORT) -> subprocess.Popen:
    proc = subprocess.Popen(
        # default bind is IPv4 0.0.0.0; split runs reach it over i6pn (IPv6)
        ["mooncake_master", "--rpc_port", str(port), "--metrics_port", "0", "--rpc_address", "::"],
        stdout=open("/tmp/mooncake_master.log", "w"),
        stderr=subprocess.STDOUT,
        env={**os.environ, "MC_USE_IPV6": "1"},
    )
    wait_port("::1", port, timeout=60)
    return proc


@dataclass
class Driver:
    """A Miles driver process; its output is echoed with a name prefix and its tail kept."""

    name: str
    argv: list[str]
    env: dict
    proc: subprocess.Popen | None = None
    lines: list[str] = field(default_factory=list)
    _cv: threading.Condition = field(default_factory=threading.Condition)

    def start(self) -> "Driver":
        self.proc = subprocess.Popen(
            self.argv,
            cwd=MILES_SRC,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._pump, daemon=True).start()
        return self

    def _pump(self) -> None:
        for line in self.proc.stdout:
            print(f"[{self.name}] {line}", end="", flush=True)
            with self._cv:
                self.lines.append(line)
                del self.lines[:-400]
                self._cv.notify_all()

    def wait_for(self, predicate, timeout: float, what: str):
        deadline = time.monotonic() + timeout
        with self._cv:
            while not (value := predicate(self)):
                if self.proc.poll() is not None:
                    raise RuntimeError(f"{self.name} exited ({self.proc.returncode}) before {what}")
                if (
                    not self._cv.wait(timeout=max(0.0, min(5.0, deadline - time.monotonic())))
                    and time.monotonic() > deadline
                ):
                    raise TimeoutError(f"{self.name}: no {what} after {timeout:.0f}s")
            return value

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def stop_app(app_id: str) -> None:
    """Stop a Modal app by id from inside a container (the request ``modal app stop`` sends)."""
    from modal._utils.async_utils import synchronizer
    from modal.client import _Client
    from modal_proto import api_pb2

    async def _stop() -> None:
        client = await _Client.from_env()
        await client._stub.AppStop(api_pb2.AppStopRequest(app_id=app_id, source=api_pb2.APP_STOP_SOURCE_CLI))

    synchronizer.create_blocking(_stop)()
