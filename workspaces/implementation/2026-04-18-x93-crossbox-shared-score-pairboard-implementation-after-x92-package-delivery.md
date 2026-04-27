# 2026-04-18 X-93 Cross-Box Shared-Score Pairboard Implementation After X-92 Package Delivery

## Question

Can the repository land one honest reusable `05-cross-box` pairboard surface for `GSA + PIA`, and does the current asset pool already support a real shared-subset read?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/crossbox_pairboard.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/cli.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_crossbox_pairboard.py`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260409-gpu-512-adaptive/scores.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/runs/gsa-loss-score-export-bounded-actual-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/cross-box/runs/crossbox-pairboard-gsa-loss-pia-actualsubset-20260418-r1/summary.json`

## Implementation Review

### 1. One reusable pairboard surface is now real

The repository now exposes one generic cross-box pairboard surface:

- internal helper: `src/diffaudit/attacks/crossbox_pairboard.py`
- CLI entry: `analyze-crossbox-pairboard`

This surface currently supports:

- plain JSON score packets with `member_scores / nonmember_scores`
- family-style nested JSON packets via `--surface-*-family`
- direct `GSA loss-score-export` summaries

The evaluator freezes one calibration/test split and emits the four planned candidate boards:

- `best_single`
- `weighted_average`
- `logistic_2feature`
- `support_disconfirm_neutral`

All four are reported with:

- `AUC`
- `ASR`
- `TPR@1%FPR`
- `TPR@0.1%FPR`

### 2. `GSA` can now participate in shared-index intersection

This implementation does more than read scalar scores.

For `GSA loss-score-export`, the loader now reads `records_path` and recovers sample IDs from filenames such as:

- `00-data_batch_1-00965.png`
- `00-data_batch_1-01278.png`

So the current blocker is no longer “`GSA` lacks scalar export”.

The stronger current truth is:

- `GSA` does have bounded scalar export
- `GSA` now also exposes enough record metadata to attempt shared-index intersection
- the remaining bottleneck is overlap size and packet matching quality, not score-surface absence

### 3. First real shared-subset read is now landed

One actual cross-box pairboard read was executed on live artifacts:

- gray-box side: `PIA 512 adaptive`
- white-box side: `GSA bounded actual loss-score export`

Resulting shared overlap:

- member overlap = `3`
- nonmember overlap = `4`
- shared member IDs = `5587 / 8553 / 9390`
- shared nonmember IDs = `4523 / 4163 / 2287 / 7311`

The emitted run summary exists at:

- `<DIFFAUDIT_ROOT>/Research/workspaces/cross-box/runs/crossbox-pairboard-gsa-loss-pia-actualsubset-20260418-r1/summary.json`

### 4. Honest reading of that first actual run

The first actual shared-subset board is **not** a `05` verdict.

Why:

1. the overlap is only `3 member + 4 nonmember`
2. the held-out test board is therefore tiny
3. the resulting perfect metrics are a consequence of tiny support, not evidence of stable low-FPR gain

So the right interpretation is narrow:

- the repo now has a real executable shared-score pairboard surface
- current admitted/bounded artifacts already support a non-zero shared subset
- but the first truthful next gate is a larger matched `GSA + PIA` packet, not a fusion promotion claim

## Verdict

- `x93_crossbox_shared_score_pairboard_implementation_verdict = positive but bounded`

More precise reading:

1. `05-cross-box` now has a reusable in-repo execution surface
2. `GSA` scalar export is no longer the main blocker
3. a real shared-subset run now exists
4. the immediate next blocker is insufficient shared overlap for an honest low-FPR verdict

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `current execution lane = R2-3 shared-score validation, now upgraded from planning-only to implementation-plus-overlap-sizing`
- `current next gate = freeze a larger matched GSA+PIA shared packet before any go/kill fusion claim`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Research/docs/comprehensive-progress.md`: update required
- `Research/docs/future-phase-e-intake.md`: update required
- `Research/docs/mainline-narrative.md`: update required
- `Platform/Runtime`: no direct handoff yet
- consumer-schema change: none
- competition/materials sync: note-level only
