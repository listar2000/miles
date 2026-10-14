# Harbor patches carried without a Harbor fork

Applied to a checkout of `HARBOR_COMMIT` (miles_ext/stack.py) when it is used.
None of these touch Harbor's installed package, so the image does not apply
them; see fork/README.md "Harbor" for when that changes.

| Patch | Applies to | Why |
| --- | --- | --- |
| `0001-swegym-adapter-pytest-rA.patch` | `adapters/swegym/` (offline task generation) | SWE-Gym repos are not in swebench's spec map, so the adapter falls back to a bare `pytest`, whose output has no per-test PASSED lines and the parser marks FAIL_TO_PASS as failed; `pytest -rA` prints them. Without it every gold patch scored 0 (modal-miles-launch RUN_LOG items 3 and 5). |

    git -C <harbor checkout at HARBOR_COMMIT> apply <this dir>/0001-swegym-adapter-pytest-rA.patch
