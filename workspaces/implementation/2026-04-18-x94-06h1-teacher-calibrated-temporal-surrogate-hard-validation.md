# 2026-04-18 X-94 06-H1 Teacher-Calibrated Temporal Surrogate Hard Validation

## Question

Can the first actual `06-H1` packet clear the `TMIA-DM-256` teacher-calibrated gates strongly enough to keep `H1 temporal QR surrogate` as the active blocker-resolution default?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\temporal_surrogate.py`
- `D:\Code\DiffAudit\Research\docs\report-bundles\gpt54\round2-results\06.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\temporal-surrogate-feature-packet-cifar10-scout64-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\tmiadm-cifar10-temporal-surrogate-teacher64-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\temporal-surrogate-eval-cifar10-teacher64-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\temporal-surrogate-feature-packet-cifar10-scout128-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\tmiadm-cifar10-temporal-surrogate-teacher128-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\temporal-surrogate-eval-cifar10-teacher128-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\temporal-surrogate-feature-packet-cifar10-teacher256-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\tmiadm-cifar10-temporal-surrogate-teacher256-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\temporal-surrogate-eval-cifar10-teacher256-20260418-r1\summary.json`

## What Was Implemented

One reusable `06-H1` CPU-first scaffold now exists in-repo:

- `export-temporal-surrogate-feature-packet`
- `evaluate-temporal-surrogate-packets`

The current packet uses:

- target-only temporal summary features on a fixed `12`-point grid
- `8` bagged linear quantile heads
- OOF isotonic calibration on the teacher packet
- frozen evaluation against `TMIA-DM long_window`

This is a real runnable bounded packet, not a planning placeholder.

## Actual Results

### 1. `64` scout

Feature export and teacher packet both succeeded on real assets.

Teacher-eval summary:

- `Spearman = 0.733753`
- `Pearson = 0.746071`
- `AUC = 0.619995`
- `teacher AUC = 0.807617`
- `threshold CV = 0.010522`

Read:

- extraction path is real
- packet is stable enough to score
- but the surrogate is still far below teacher quality

### 2. `128` scout

The same fixed packet improved directionally:

- `Spearman = 0.737358`
- `Pearson = 0.771908`
- `AUC = 0.676758`
- `teacher AUC = 0.839539`
- `TPR@1%FPR = 0.070312`
- `teacher TPR@1%FPR = 0.03125`
- `threshold CV = 0.010201`

Read:

- the packet is not random noise
- low-FPR tail can occasionally look competitive
- but correlation and overall fidelity are still below the intended teacher-gated bar

### 3. `256` hard-validation rung

The first real hard-validation rung still fails the key teacher gates:

- `Spearman = 0.748677`
- `Pearson = 0.790525`
- `AUC = 0.687477`
- `teacher AUC = 0.850357`
- `AUC delta vs teacher = -0.162880`
- `TPR@1%FPR = 0.007812`
- `teacher TPR@1%FPR = 0.023438`
- `threshold CV = 0.009821`

Gate read:

- `spearman_ge_0_8 = false`
- `pearson_ge_0_8 = false`
- `auc_delta_abs_le_0_05 = false`
- `tpr_at_1pct_fpr_delta_abs_le_0_05 = true`
- `threshold_cv_lt_0_15 = true`

## Verdict

- `x94_06h1_teacher_calibrated_temporal_surrogate_hard_validation_verdict = fallback-required`

More precise reading:

1. `06-H1` now has a real executable packet and real bounded evidence
2. the packet is stable enough to evaluate and not degenerate
3. but the first fixed `H1` instantiation does **not** clear the `256` teacher gates
4. so the next honest move is `06-H2 RMIA / BASE temporal LR`, not `512` frozen transfer

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `current execution lane = 06 fallback to H2 after H1 hard-validation miss`
- `do not run 512 frozen transfer on this H1 packet`

## Canonical Evidence Anchor

Primary anchor:

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\temporal-surrogate-eval-cifar10-teacher256-20260418-r1\summary.json`

Supporting anchors:

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\temporal-surrogate-eval-cifar10-teacher64-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\temporal-surrogate-eval-cifar10-teacher128-20260418-r1\summary.json`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Research/docs/comprehensive-progress.md`: update required
- `Research/docs/future-phase-e-intake.md`: update required
- `Research/docs/mainline-narrative.md`: update required
- `Research/docs/research-autonomous-execution-prompt.md`: update required
- `Research/docs/codex-roadmap-execution-prompt.md`: update required
- `Platform/Runtime`: no direct handoff yet
- `competition-materials`: note-level only
