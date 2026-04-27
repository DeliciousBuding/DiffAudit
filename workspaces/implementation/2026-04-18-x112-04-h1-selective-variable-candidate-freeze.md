# 2026-04-18 X-112 04 H1 Selective-Variable Candidate Freeze

## Question

After `X-110` ruled out another immediate pressure-based rerun, which single selective variable is the next honest same-family candidate inside `04-H1`, if the lane is reopened?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-18-x110-04-h1-post-alpha-parameter-selection-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-pilot-k16-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-pilot-k16-alpha075-20260418-r1/summary.json`
- local schedule sanity check under Python `random.Random(0)` for `32` Bernoulli branch draws

## What Was Resolved

### 1. The next same-family axis should be frequency, not pressure

This is already fixed by `X-109` and `X-110`:

1. stronger forget pressure failed
2. another immediate `k` move is not honest
3. same-family continuation, if any, should now reduce forget-branch frequency rather than amplify the forget term

### 2. One concrete mid-strength candidate is now frozen

Under Python `random.Random(0)` and a `32`-step branch schedule:

- `mixture_lambda = 0.5` yields about `12` forget branches
- `mixture_lambda = 0.4375` yields about `9`
- `mixture_lambda = 0.375` yields about `6`
- `mixture_lambda = 0.25` collapses to about `1`

So the first honest selective-frequency candidate is:

- `k16`
- `alpha = 0.5`
- `mixture_lambda = 0.4375`
- `32` steps

because it is the first rung that:

1. materially lowers forget exposure from the current baseline
2. does not immediately collapse toward near-no-op territory
3. keeps the rest of the contract unchanged

## Verdict

- `x112_04_h1_selective_variable_candidate = k16 mixture_lambda_down to 0.4375`

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = 04-H1 k16 mixture_lambda-down pilot (mixture_lambda = 0.4375, alpha = 0.5, 32 steps)`

Boundary:

1. this is only a conditional candidate freeze
2. it is not a released GPU question yet
3. if `04` yields to another lane, this candidate remains parked rather than auto-executed

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-18-x112-04-h1-selective-variable-candidate-freeze.md`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Research/docs/comprehensive-progress.md`: update optional
- `Research/docs/mainline-narrative.md`: update optional
- `Platform/Runtime`: no schema change required
