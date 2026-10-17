"""Training-task filter sweep: one Sail sandbox per task checks three filters at once.

In each trial, in training order:

1. agent setup: install mini-swe-agent (the training harness) the way Harbor
   does, with public network for the install only, and record whether it
   worked (``probe_agent.ProbeOracleAgent``; it never fails the trial);
2. agent run: the task's reference solution (Harbor's oracle);
3. verifier: the task's tests, timed.

``tasks`` lists a platform taskset into the sweep's input and drops GPU tasks
(filter 1) from ``task.toml``; ``sweep`` is the Modal app that fetches each task
version from the Mobius platform (``mobius``), runs the trial on our Sail
provider (``miles_ext.harbor.sail``) and writes one ``summary.summarize`` row
per task to the ``miles-fork-results`` Volume; ``final`` merges the runs into
verdicts and keep lists with ``summary.reasons``, the one owner of the drop rules.
"""
