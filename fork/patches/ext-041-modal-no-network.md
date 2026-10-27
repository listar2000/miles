# ext-041: Harbor's Modal sandboxes run no-network tasks

## Why
On Modal, every trial of a no-network task failed in agent setup with "sandbox was created without a domain
allowlist; enabling domain filtering while running is not currently supported" (sandbox benchmark smoke run
sbxbench-1027-a5, all Modal noop and read trials). A no-network task changes its network policy during a trial:
`SandboxMiniSweAgent.setup` opens public network for the harness install, and the agent phase allows the session
tunnel's host. The pinned Harbor (917b3c1) sees the change and creates the sandbox with the task's own policy as
its filter. For no network that is an empty domain list, which Modal reads as "no filter", and Modal refuses
to add a filter later. Sail switches policies at run time, so Sail runs never hit it. Public tasks never
switch.

## Decision
- `miles_ext/harbor/modal_env.py:ModalEnvironment` subclasses Harbor's Modal provider and overrides only
  `_create_sandbox`. A no-network sandbox with a policy change ahead is created with the allow-all filter (`*`,
  which Modal can change later) and narrowed to no network before the call returns. Only the keep-alive
  entrypoint runs in between. If narrowing fails, the sandbox is terminated. Every other sandbox is created
  exactly as upstream creates it.
- Selected wherever a run names Harbor's `modal` type: `agent_function._provider_config` swaps the import path
  into upstream's config and keeps every other field (resource policy, overrides, kwargs). ext-040's profiler
  wraps this class too (`sandbox_profile.inner_import_path`). Rejected: making no-network tasks public in the
  benchmark, because training on Modal would still fail on them.

## How it is applied
- New `miles_ext/harbor/modal_env.py`; `agent_function._provider_config` and
  `sandbox_profile.inner_import_path` read `modal_env.REPLACES_TYPE` / `IMPORT_PATH`.
- Check: `python -m pytest miles_ext/tests/test_modal_env.py miles_ext/tests/test_sandbox_profile.py`.
- On a Harbor bump: `_create_sandbox`, `_dynamic_network`, `_dynamic_network_kwargs` and
  `_experimental_set_outbound_network_policy` are what this relies on.

## Remove when
Harbor's Modal provider creates a dynamic-policy sandbox with a filter Modal can change later. Upstream status:
not filed.
