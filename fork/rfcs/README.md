# RFCs

Proposals for changes to how this repository works (branches, versions,
builds, names), written before the change and kept after it. Patch pages
(`fork/patches/`) explain one change in the code; an RFC decides a process
or a design that several changes follow.

- **File:** `fork/rfcs/NNNN-<slug>.md`, numbered in order, plus one row below.
- **Status line** under the title: draft, accepted (date, what is done) or
  replaced (by which RFC). A replaced RFC stays, unedited apart from its status.
- **Content:** what is decided and by whom, the design, what it conflicts
  with, and the order of steps, marking the ones that need the owner's OK.

| RFC | Status |
| --- | --- |
| [0001 One running main, latest or a stable version](0001-main-and-stable-versions.md) | accepted 2026-10-15; in the rulebook |
| [0002 SFT data tokenized the way the model sees it when served](0002-tito-aware-sft-tokenization.md) | accepted 2026-11-01; implemented (ext-048, hook-017) |
| [0003 Connectors from other data formats to SFT records, with Harbor ATIF first](0003-sft-record-connectors.md) | accepted 2026-11-01; implemented (ext-049) |
| [0004 Engines serve the trainer's own FP8 bytes, so the first GPU-delta sync is empty](0004-trainer-format-gpu-delta-base.md) | accepted 2026-11-03; base built for GLM-5.3-Flash, launcher step to build |
