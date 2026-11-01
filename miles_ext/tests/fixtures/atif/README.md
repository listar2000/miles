# ATIF trial directories for the SFT connector tests

Byte-for-byte copies from [harbor-framework/harbor](https://github.com/harbor-framework/harbor)
at commit `ac533ec72bc506c8a6bf6f296ba6ab80707fad89` (Apache-2.0), placed in the layout of a
Harbor trial directory (`<trial>/agent/trajectory*.json`). Do not edit them; copy a newer
version instead.

| Trial directory | Harbor source |
| --- | --- |
| `terminus2-summarization/agent/trajectory.json` | `tests/golden/terminus_2/hello-world-context-summarization.trajectory.json` |
| `terminus2-summarization/agent/trajectory.summarization-1-{summary,questions,answers}.json` | `tests/golden/terminus_2/hello-world-context-summarization.trajectory.summarization-1-{summary,questions,answers}.json` |
| `terminus2-linear-history/agent/trajectory.json` | `tests/golden/terminus_2/hello-world-context-summarization-linear-history.trajectory.json` |
| `terminus2-linear-history/agent/trajectory.cont-1.json` | `tests/golden/terminus_2/hello-world-context-summarization-linear-history.trajectory.cont-1.json` |
| `terminus2-invalid-json/agent/trajectory.json` | `tests/golden/terminus_2/hello-world-invalid-json.trajectory.json` |
| `terminus2-timeout/agent/trajectory.json` | `tests/golden/terminus_2/hello-world-timeout.trajectory.json` |
| `claude-code-hello/agent/trajectory.json` | `examples/tasks/hello-load-atif-trajectory-task-level/trajectory.json` |

The linear-history golden references summarization files that Harbor's test data does not
include, so that trial also exercises a missing subagent.
