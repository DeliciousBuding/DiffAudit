# 2026-04-18 X-103 04 H1 Forgotten-Subset First Review

## Question

After `X-102` produced the first actual `k32` retain+forget checkpoint, does the first attached attack-side subset review show any encouraging privacy signal on the forgotten members?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/defenses/risk_targeted_unlearning.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/cli.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_risk_targeted_unlearning.py`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-pilot-k32-20260418-r2/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/runs/gsa-loss-score-export-targeted-full-overlap-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-k32-20260418-r1/summary.json`

## What Landed

### 1. One reusable subset-review surface now exists

The repo now exposes `review-risk-targeted-unlearning-pilot`.

It does one narrow thing:

1. borrow shadow exports from an existing undefended `GSA loss-score-export` packet
2. rerun only the target subset on:
   - baseline checkpoint
   - defended checkpoint
3. evaluate both with the same transferred shadow threshold
4. write baseline/defended summaries plus transfer deltas into one machine-readable review packet

This is intentionally a **defense-unaware diagnostic** rather than a defense-aware verdict.

### 2. One first real forgotten-subset review now exists

The first actual review used:

- subset = exported `k32` forgotten members + matched nonmembers
- allowlist size = `64` unique sample ids
- actual exported target files:
  - baseline member = `36`
  - baseline nonmember = `36`
  - defended member = `36`
  - defended nonmember = `36`
- shadow threshold source = current full-overlap undefended packet

## Actual Read

### Baseline vs defended target-transfer metrics

On this first diagnostic:

- baseline:
  - `AUC = 0.774691`
  - `ASR = 0.513889`
  - `TPR@1%FPR = 0.222222`
  - `TPR@0.1%FPR = 0.222222`
- defended:
  - `AUC = 0.755401`
  - `ASR = 0.541667`
  - `TPR@1%FPR = 0.027778`
  - `TPR@0.1%FPR = 0.027778`

Transfer deltas (`defended - baseline`):

- `AUC = -0.01929`
- `ASR = +0.027778`
- `TPR@1%FPR = -0.194444`
- `TPR@0.1%FPR = -0.194444`

### Honest reading

This is not an encouraging first read.

The strongest honest interpretation is:

1. there is now one real attack-side diagnostic attached to `04-H1`
2. under that diagnostic, the defended checkpoint does **not** improve the forgotten-subset low-FPR story
3. the first attached review is therefore negative, especially on low-FPR tails

But the lane should not be killed yet, because this review is still narrow:

1. it borrows undefended shadows
2. it only covers forgotten members + matched controls
3. it does not yet include retained subset or full split
4. it is not defense-aware retraining

## Verdict

- `x103_04_h1_first_subset_review_verdict = negative but provisional`

More precise reading:

1. `04-H1` now has both an actual pilot and an actual attack-side read
2. the first read is unfavorable, so the lane no longer deserves optimistic wording
3. but the read is not yet strong enough to close the family, because the contract is still defense-unaware and subset-only

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `04 current state = actual pilot exists, first forgotten-subset diagnostic is negative`
- `next honest move = retained-subset / full-split diagnostic before any defense-aware rerun`

## Canonical Evidence Anchor

Primary anchor:

- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-k32-20260418-r1/summary.json`

Supporting anchors:

- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-pilot-k32-20260418-r2/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/runs/gsa-loss-score-export-targeted-full-overlap-20260418-r1/summary.json`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Research/docs/comprehensive-progress.md`: update required
- `Research/docs/mainline-narrative.md`: update required
- `Research/docs/future-phase-e-intake.md`: update required
- `Platform/Runtime`: no direct handoff yet
