# 2026-04-18 X-95 06-H2 Temporal LR Fallback Calibration Review

## Question

After `06-H1` missed the `256` teacher gates, can the first fixed `06-H2 RMIA / BASE temporal LR` packet provide a cleaner blocker-resolution fallback?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\temporal_lr.py`
- `D:\Code\DiffAudit\Research\docs\report-bundles\gpt54\round2-results\06.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\temporal-surrogate-feature-packet-cifar10-teacher256-20260418-r1\feature-packet.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\temporal-lr-eval-cifar10-cal256-20260418-r1\summary.json`

## What Was Implemented

One reusable `H2` fallback evaluator now exists in-repo:

- `evaluate-temporal-lr-packets`

The first bounded packet keeps the fallback intentionally simple:

- primary temporal statistic: `eps_abs_mean_late`
- sensitivity statistic: `eps_abs_late_over_early`
- univariate Gaussian likelihood-ratio scoring
- repeated OOF calibration on the local packet

This is narrower than `H1` on purpose. It tests whether a simpler temporal LR family is more honest and more stable than the multi-feature surrogate.

## Actual Result

Calibration packet:

- `256 / 256`
- source packet: `temporal-surrogate-feature-packet-cifar10-teacher256-20260418-r1`

Primary candidate `eps_abs_mean_late`:

- `AUC = 0.644142`
- `ASR = 0.617188`
- `TPR@1%FPR = 0.007812`
- `TPR@0.1%FPR = 0.0`
- `threshold_cv = 0.806137`

Sensitivity candidate `eps_abs_late_over_early`:

- `AUC = 0.637344`
- `ASR = 0.611328`
- `TPR@1%FPR = 0.015625`
- `TPR@0.1%FPR = 0.0`
- `threshold_cv = 0.862850`

## Reading

This packet is weaker than the already-missed `H1` packet on the calibration board.

The key problem is not just `AUC`.

The stronger problem is:

- low-FPR behavior is weak
- threshold stability is extremely poor
- the sensitivity statistic does not rescue the primary statistic

So this is not an honest `256 -> 512` transfer candidate.

## Verdict

- `x95_06h2_temporal_lr_fallback_calibration_review_verdict = negative but useful`

More precise reading:

1. the repo now has a real executable `H2` fallback surface
2. the first fixed `H2` packet was tested on real assets
3. that packet is too weak and too unstable to justify `512` frozen transfer
4. the next honest move is no longer another immediate `H2` transfer run

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `current 06 read = per-sample blocker-resolution paths H1 and first fixed H2 both miss`
- `current next move = governance fallback review on H5 or yield 06 main slot to 05`

## Canonical Evidence Anchor

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\temporal-lr-eval-cifar10-cal256-20260418-r1\summary.json`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Research/docs/comprehensive-progress.md`: update required
- `Research/docs/future-phase-e-intake.md`: update required
- `Research/docs/mainline-narrative.md`: update required
- `Research/docs/research-autonomous-execution-prompt.md`: update required
- `Research/docs/codex-roadmap-execution-prompt.md`: update required
- `Platform/Runtime`: no direct handoff yet
