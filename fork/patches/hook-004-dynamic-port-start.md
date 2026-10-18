# hook-004: `MILES_DYNAMIC_PORT_START`, disjoint port ranges for deployments sharing a host

## Why
Each split-run deployment (primary, inference, trainer) has its own `PortAllocator`, and every
allocator starts at 20000. When two deployments share a host (a one-container smoke test, or the
rollout app that holds the primary and the engines), both hand out 20001: the primary's
InferenceController and the SGLang engine took the same port, and the driver's
`/freeze_gc` request hit the wrong server (404). The free-port check cannot catch it: both
allocators pick the port before either binds it.

## Decision
Read the start of the dynamic range from env `MILES_DYNAMIC_PORT_START` (default 20000, unchanged).
The ext-021 launcher gives each deployment on a host its own block (20000, 24000, 28000), below
Ray's ports (10000-18999 in that launcher) and the kernel's ephemeral range.
Rejected: a dual-stack bind in `is_port_available` (a probe showed the IPv4 bind already fails
against a `[::]` listener, so the check was not the gap); starting deployments strictly one after
another (still races with engines that start later).

## How it is applied
`miles/utils/workers/addr_allocator.py`, one line plus the `os` import, marked `[fork hook-004]`.
Verify: in a one-container split run the inference controller and the engine get different ports.
On a sync, keep the env read if upstream moves the constant.

## Remove when
Upstream makes the dynamic port range configurable per deployment. Status: not filed.
