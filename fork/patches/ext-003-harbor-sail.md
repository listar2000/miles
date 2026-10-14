# ext-003: Sail sandboxes from Harbor's own Sail provider

Trials can run on [Sail](https://docs.sailresearch.com/sailboxes) microVMs
instead of Modal Sandboxes: `sandbox.provider: sail` in a recipe. Kind: Ext,
with one vendored file. Commits: infra-002 (vendored-file check), infra-004
(Sail SDK in the deps layer), ext-003 (the provider).

## Why

Sail support is a dependency-of-a-dependency gap. Miles v0.1.2 runs trials
through Harbor, pinned to the branch Miles documents
(`harbor-miles-v0.20.0` @ `917b3c1b`, 2026-10-02). Harbor added Sail on main
four days later ([#3475](https://github.com/harbor-framework/harbor/pull/3475),
commit `e2f6db74`): `environments/sail.py`, an `EnvironmentType.SAIL` member,
a factory entry and a `sail` extra. None of it is at our pin, and the branch
head has not moved since.

Miles adds a second gap. Its agent function builds the trial environment
only from `HARBOR_ENV_TYPE` via `EnvironmentType(raw)`, so `sail` raises
before Harbor is reached. Its credentials table has no Sail entry, but none
is needed: the Sail SDK reads `SAIL_API_KEY` from the worker environment,
which the launcher's Modal Secret provides.

## Decision

We carry Harbor's `sail.py` verbatim in `miles_ext/` and load it through
Harbor's `EnvironmentConfig.import_path`, which is Harbor's documented seam for
environments it doesn't ship. We don't fork Harbor or change upstream Miles
files.

The evidence that the newer file runs on the older Harbor: everything it
imports exists at the pin. Harbor's own `test_sail.py` (76 tests) run against
the vendored file on the pinned Harbor gives 51 failures, and all of them are
`EnvironmentType has no attribute SAIL`. With `type()` returning `"sail"` in
a subclass, all 76 pass.

| Option | Verdict |
| --- | --- |
| **Vendor Harbor main's `sail.py` verbatim, subclass it** | **Chosen.** This is the code Harbor maintains, checked by Harbor's own tests on our pin. Removing it later is a delete once the pinned branch has it. |
| Sail SDK's own adapter (`sail.harbor:SailboxEnvironment`) | Copies no code, but the adapter is Sail's, it is versioned with the SDK, and it behaves differently: type `sailbox`, its app comes from a global `SAIL_APP`, and `stop` sleeps the box. It will never converge with Harbor's provider. The lifetime cap would mean overriding its much larger `start()`. |
| Fork Harbor and cherry-pick `e2f6db74` | A second repository to rebase on every sync, for a change that needs no edit to installed Harbor code. The rule (fork/README.md "Harbor") keeps a Harbor fork for that case. |
| Move the Harbor pin to main | That is 446 commits past the branch Miles documents and tests, and it drops what Miles needs from that branch (terminus-2 response-length policy, `MaxSeqLenExceededError`). |
| A Hook in Miles' `_environment_config` (see below) | Cleaner for the Miles side, but Hooks go before Infra and Ext in the series. Adding one now would rewrite history under the released `fork-v0.1.2-r1` tag. |

Two smaller decisions:

- **Caps.** Neither Harbor's provider nor Sail's sets a lifetime, and Sail's
  default auto-sleep fires after 30 s idle. The subclass passes
  `max_lifetime_seconds` and `AutoSleep.not_before(idle)` from the recipe's
  `sandbox.lifetime_secs` and `sandbox.idle_secs` (7200 and 900, the Modal
  values). Sail enforces the lifetime server-side, so a Sailbox dies even if
  the trainer that made it is killed.
- **Miles' agent function.** `miles_ext/harbor/agent_function.py` replaces
  `_environment_config` on the loaded upstream module. The replacement honours
  `HARBOR_ENV_IMPORT_PATH` and delegates unchanged otherwise. This is the one
  place the fork reaches into upstream module state. `upstream.py` lists the
  name in its API check, so a rename fails at import. On the next sync,
  convert it to a Hook (`HARBOR_ENV_IMPORT_PATH` in `_environment_config`, a
  few lines). It is a candidate to send upstream.

## How it is applied

| File | Role |
| --- | --- |
| `miles_ext/harbor/sail/harbor_sail.py` | Harbor's `src/harbor/environments/sail.py` @ `e2f6db74`, byte-identical (blob `750e9006`, entry in `fork/vendored.json`) |
| `miles_ext/harbor/sail/__init__.py` | `SailEnvironment(harbor_sail.SailEnvironment)`: `type()` returns `"sail"`, the caps (an override of `_create_sailbox` that adds only `**caps` to the create call), and `terminate_app_sailboxes(app)` |
| `miles_ext/harbor/agent_function.py`, `upstream.py` | the `HARBOR_ENV_IMPORT_PATH` environment config |
| `miles_ext/recipes/recipe.py` | `sandbox.provider` is `modal` or `sail`. For Sail it sets `HARBOR_ENV_IMPORT_PATH` and kwargs `{app_name, max_lifetime_secs, idle_secs}`, and sends no resource policy (Sail sizes are fixed and billed by observed usage; a `limit` policy would fail every trial) |
| `miles_ext/launch/modal_app.py` | at exit, terminates every Sailbox in the run's app `miles-fork-sbx-<run_id>`, sleeping ones included; the dry check runs the provider's preflight and resource-policy validation |
| `miles_ext/stack.py`, `fork/docker/build.py` | one deps install line for both image paths, now with `sail==0.13.0`. The pin is exact because the vendored file imports SDK internals (`sail._core_client`, `sail.image`) |

**Image.** This needs a new deps layer. `fork-v0.1.2-r1` has no Sail SDK, so
it can't run a Sail recipe. Use a release built after infra-004, or
`MILES_FORK_COMMIT=<sha>` for testing.

**Verify.**

```bash
python fork/tools/ledger.py check          # includes the vendored blob check
MILES_RECIPE=miles_ext/recipes/configs/qwen35-9b-swegym8-sail.yaml \
  modal run -m miles_ext.launch.modal_app --run-id <id> --check-only
```

To rerun Harbor's tests against the subclass, take Harbor's
`tests/unit/environments/test_sail.py` at the vendored commit and point its
imports at `miles_ext.harbor.sail.harbor_sail` (module) and
`miles_ext.harbor.sail.SailEnvironment` (class). They need Harbor at
`HARBOR_COMMIT`, `sail==0.13.0`, pytest-asyncio, and Harbor's `temp_dir`
fixture.

Validated 2026-10-14. All 8 SWE-Gym training tasks score 1.0 with Harbor's
oracle agent on Sail. The 10-step Qwen3.5-9B run on Sail (W&B `30cpjc0l`, in
`miles-swe-rl` from before the per-campaign projects) finished 844 trials with
no infrastructure failures. Its mean reward over 10 steps was 0.26, against
0.28 for the Modal baseline `ezq40tsy` over the same steps, and its step time
matched. All 896 Sailboxes it created ended terminated. The launch repo's
`run-logs/2026-10-14-sail-10step.md` has the details.

**Refresh** (Harbor changes `sail.py` on main):

1. Copy the new file over `harbor_sail.py`.
2. Update `commit` and `blob` in `fork/vendored.json`.
3. Re-diff `_create_sailbox` against the override in `__init__.py`.
4. Rerun Harbor's tests as above.

## Remove when

The pinned `harbor-miles-*` branch contains `environments/sail.py`. Then:

1. Delete `harbor_sail.py` and its `vendored.json` entry.
2. Subclass `harbor.environments.sail.SailEnvironment` instead, and drop the
   `type()` override.
3. Keep the caps override until Harbor's provider accepts a lifetime and an
   auto-sleep window.

If Miles' agent function by then honours an environment import path, or
accepts `HARBOR_ENV_TYPE=sail`, also drop the `_environment_config`
replacement.
