# Getting help and the responsibility boundary

This repo is a training framework. The people and agents who run training with it
are its users. The maintainers own the framework underneath.

| Users own | Maintainers own |
| --- | --- |
| recipes and their values; task sets and data prep; rewards (task verifiers); harness choice; algorithm code added through the extension points ([extending.md](extending.md)); their own run budgets, schedules, W&B projects and Modal names | upstream Miles and SGLang fixes and hooks, kernels, rollout and training speed, memory and parallel layouts, weight sync, the image and stable versions, the launcher and `miles_ext.platform`, and the extension points themselves |

If what you need sits on the right-hand side, don't patch it locally or in a
recipe. Open an issue instead, so the fix lands once for everyone.

## Opening an issue

Use the form that fits at [New issue](https://github.com/listar2000/miles/issues/new/choose):

- **Run failure:** something broke or misbehaved in a run. Attach the trainer
  version (or image digest), the recipe or its diff from a shipped one, the run id,
  the first error lines from the app log, and the W&B run if there is one.
- **Extension point or feature:** you want to plug something in and the current
  points don't fit, or you need a capability the fork lacks. Say what you are
  trying to do, what you tried, and why it doesn't fit.
- **Question:** anything else.

New issues get `needs-triage`. A maintainer replaces it with an area label
(`area:rollout`, `area:trainer`, `area:launch`, `area:recipes`, `area:harbor`)
when the issue is picked up. The fix links back to the issue.

## Sending a PR

Extensions under `miles_ext/` are welcome when they are general, meaning other
users would want them too. Open the PR as a draft, follow
[fork/README.md](../README.md), and run the
[miles-review checklist](../../.claude/skills/miles-review/SKILL.md) on it
yourself before asking for review. A maintainer merges.
