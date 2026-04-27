# 2026-04-18 X-106 04 H1 K16 Changed-Pilot Tri-Board Review

## Question

After `X-105` ruled out spending more on the current `k32` instantiation, does the smallest changed pilot (`k16`, same hyperparameters) produce a more honest `04-H1` working instantiation?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-prep-full-overlap-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-pilot-k16-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-retained-companion-k16-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-k16-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-retained-k16-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-fullsplit-k16-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-k32-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-retained-k32-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-fullsplit-k32-pairednoise-20260418-r1/summary.json`

## What Landed

### 1. One first changed pilot now exists

The repo now contains a second real `04-H1` pilot:

- `k = 16`
- same target checkpoint
- same `32` training steps
- same `alpha = 0.5`
- same `mixture_lambda = 0.5`
- same `lr = 1e-5`

This is the smallest honest change after `X-105`:

1. same family
2. same optimizer/training contract
3. narrower forget set only

### 2. One full paired-noise tri-board now exists for `k16`

`k16` now has:

1. forgotten subset board
2. retained high-risk companion board
3. full-split board

So it can be compared to `k32` under the same `paired-noise` review surface rather than by mixed review protocols.

## Actual Read

### `k16` paired-noise target-transfer metrics

Forgotten subset:

- baseline:
  - `AUC = 0.903509`
  - `ASR = 0.513514`
  - `TPR@1%FPR = 0.315789`
  - `TPR@0.1%FPR = 0.315789`
- defended:
  - `AUC = 0.885965`
  - `ASR = 0.540541`
  - `TPR@1%FPR = 0.368421`
  - `TPR@0.1%FPR = 0.368421`

Retained companion:

- baseline:
  - `AUC = 0.781046`
  - `ASR = 0.514286`
  - `TPR@1%FPR = 0.235294`
  - `TPR@0.1%FPR = 0.235294`
- defended:
  - `AUC = 0.781046`
  - `ASR = 0.571429`
  - `TPR@1%FPR = 0.294118`
  - `TPR@0.1%FPR = 0.294118`

Full split:

- baseline:
  - `AUC = 0.623331`
  - `ASR = 0.5585`
  - `TPR@1%FPR = 0.027`
  - `TPR@0.1%FPR = 0.002`
- defended:
  - `AUC = 0.622141`
  - `ASR = 0.5675`
  - `TPR@1%FPR = 0.026`
  - `TPR@0.1%FPR = 0.002`

### Comparison to `k32`

Relative to `k32` paired-noise:

1. forgotten subset:
   - `k32` was negative on both `AUC` and low-FPR tails
   - `k16` still loses `AUC`, but both low-FPR tails now improve
2. retained companion:
   - `k32` was weak/flat
   - `k16` stays flat on `AUC` and improves both low-FPR tails
3. full split:
   - `k32` was slightly negative across `AUC / ASR / TPR@1%FPR`
   - `k16` is much closer to neutral: `AUC delta = -0.00119`, `TPR@1%FPR delta = -0.001`, `TPR@0.1%FPR delta = 0`

### Shift read

On the paired-noise full-split board for `k16`:

- all members mean loss shift = `+0.001841`
- forgotten members mean loss shift = `+0.002674`
- all nonmembers mean loss shift = `+0.001753`

This is still not a clean, large forgotten-only effect.

But it is much better behaved than `k32`:

1. global drift is much smaller
2. forgotten members do move slightly more than the global background
3. target-wide separation is nearly neutral rather than clearly worse

## Verdict

- `x106_04_h1_k16_changed_pilot_verdict = better than k32, still not defense-positive`

More precise reading:

1. `k16` is the first changed pilot that materially improves the `04-H1` picture
2. it does not justify defense-positive wording
3. but it does justify replacing `k32` as the current working instantiation

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = bounded changed-pilot follow-up only`
- `04 current state = k16 replaces k32 as current working instantiation`

Practical read:

1. do not spend defense-aware rerun cost on `k32`
2. keep `k16` alive as the current best `04-H1` instantiation
3. the next honest move inside `04-H1` is one more bounded changed pilot around the `k16` regime, not a family switch and not a defense-aware shadow retrain yet

## Canonical Evidence Anchor

Primary anchors:

- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-pilot-k16-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-k16-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-retained-k16-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-fullsplit-k16-pairednoise-20260418-r1/summary.json`

Supporting anchors:

- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-k32-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-retained-k32-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-fullsplit-k32-pairednoise-20260418-r1/summary.json`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Research/docs/comprehensive-progress.md`: update required
- `Research/docs/mainline-narrative.md`: update required
- `Research/docs/future-phase-e-intake.md`: update required
- `Platform/Runtime`: still no direct handoff yet
