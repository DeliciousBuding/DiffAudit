# 2026-04-21 X-123 R2-5b PIA + SimA Support-Fusion Contract Review

## Question

Given current repo truth, is `PIA + SimA` support-fusion / calibration review already executable on an honest shared packet, or is another contract step still missing?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/sima_adapter.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/pia_adapter.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/crossbox_pairboard.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_sima_adapter.py`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-graybox-sima-feasibility-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-graybox-sima-rescan-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/sima-cifar10-runtime-feasibility-20260416-cpu-32-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/sima-cifar10-runtime-rescan-20260416-cpu-32-r2/summary.json`
- `<DIFFAUDIT_ROOT>/Research/docs/report-bundles/gpt54/round2-results/02.md`

## Findings

### 1. The generic fusion evaluator is already real

The repo already has a reusable two-surface evaluator:

- `src/diffaudit/attacks/crossbox_pairboard.py`

It can evaluate:

- `best_single`
- `weighted_average`
- `logistic_2feature`
- `support_disconfirm_neutral`

on any two surfaces that emit:

- `member_scores`
- `nonmember_scores`
- optional shared `member_indices / nonmember_indices`

### 2. `PIA` is already contract-compatible

`PIA` already exposes pairboard-ready packet export:

- `pia-packet-score-export`
- `member_scores / nonmember_scores`
- exact-index packet control

So `PIA` is not the blocker.

### 3. `SimA` is execution-feasible but not yet fusion-contract-compatible

`SimA` already has:

- bounded runtime feasibility implementation
- local tests
- two bounded CPU packets

But current `SimA` artifacts still stop at:

- aggregate summary metrics
- timestep scan table

They do **not** emit the pairboard-required per-sample surface:

- no `member_scores` JSON payload
- no `nonmember_scores` JSON payload
- no exact `member_indices / nonmember_indices`
- no packet export mode aligned with `PIA`

### 4. Therefore `PIA + SimA` fusion review is not the immediate next executable step

The honest next step is **not** to force a calibration/fusion board yet.

The honest next step is to freeze a `SimA` packet-score export contract first.

## Verdict

`blocked but useful`.

Sharper control truth:

1. `PIA + SimA` remains a valid support-lane direction
2. but current fusion review is still blocked on `SimA` packet-surface compatibility
3. the next CPU-first task should therefore be:
   - `R2-5c SimA packet-score export contract review`

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-21-x123-r25b-pia-sima-support-fusion-contract-review.md`

## Handoff

- `Research/ROADMAP.md`: yes
- `docs/comprehensive-progress.md`: yes
- `docs/research-autonomous-execution-prompt.md`: yes
- `docs/codex-roadmap-execution-prompt.md`: yes
- `workspaces/implementation/challenger-queue.md`: yes
- `Platform/Runtime`: no

Reason:

This is a research-side support-lane contract decision only. No admitted table, Runtime endpoint, or Platform schema changes are justified.
