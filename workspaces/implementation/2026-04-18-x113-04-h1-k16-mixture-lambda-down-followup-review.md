# 2026-04-18 X-113 04 H1 K16 Mixture-Lambda-Down Follow-Up Review

## Question

After `X-112` froze `k16 + mixture_lambda-down (0.4375)` as the first conditional selective-variable candidate, does reducing forget-branch frequency improve the current `04-H1` working instantiation?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-18-x112-04-h1-selective-variable-candidate-freeze.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-pilot-k16-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-pilot-k16-lambda04375-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-k16-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-k16-lambda04375-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-retained-k16-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-retained-k16-lambda04375-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-fullsplit-k16-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-fullsplit-k16-lambda04375-pairednoise-20260418-r1/summary.json`

## What Landed

### 1. One real `k16 mixture_lambda-down` pilot now exists

The repo now contains one actual changed pilot with:

- `k = 16`
- `alpha = 0.5`
- `mixture_lambda = 0.4375`
- `32` steps

### 2. One full paired-noise tri-board also exists for the lambda-down rung

The new rung now has:

1. forgotten subset board
2. retained high-risk companion board
3. full-split board

## Actual Read

### Pilot-side training read

Original `k16`:

- `branch_counts = 19 / 13`
- `mean_keep_loss = 0.024214`
- `mean_forget_loss = 0.022605`
- `mean_objective = 0.019623`

`k16 mixture_lambda-down`:

- `branch_counts = 19 / 13`
- `mean_keep_loss = 0.027793`
- `mean_forget_loss = 0.019465`
- `mean_objective = 0.023840`

Read:

1. the realized branch schedule did not actually become sparser on this run
2. keep-side loss increased
3. this already weakens the case for promotion before looking at attack-side boards

### Forgotten subset

Old `k16`:

- `AUC 0.903509 -> 0.885965`
- `TPR@1%FPR 0.315789 -> 0.368421`
- `TPR@0.1%FPR 0.315789 -> 0.368421`

`k16 mixture_lambda-down`:

- `AUC 0.903509 -> 0.885965`
- `TPR@1%FPR 0.315789 -> 0.263158`
- `TPR@0.1%FPR 0.315789 -> 0.263158`

Read:

1. `AUC` does not improve
2. both forgotten low-FPR tails get worse
3. this loses the main reason `k16` stayed alive

### Retained companion

Old `k16`:

- `AUC 0.781046 -> 0.781046`
- `TPR@1%FPR 0.235294 -> 0.294118`
- `TPR@0.1%FPR 0.235294 -> 0.294118`

`k16 mixture_lambda-down`:

- `AUC 0.781046 -> 0.787582`
- `TPR@1%FPR 0.235294 -> 0.176471`
- `TPR@0.1%FPR 0.235294 -> 0.176471`

Read:

1. `AUC` rises slightly
2. but both low-FPR tails regress sharply
3. under the current `04` gate, this is not a real improvement

### Full split

Old `k16`:

- `AUC 0.623331 -> 0.622141`
- `ASR 0.5585 -> 0.5675`
- `TPR@1%FPR 0.027 -> 0.026`
- `TPR@0.1%FPR 0.002 -> 0.002`

`k16 mixture_lambda-down`:

- `AUC 0.623331 -> 0.624224`
- `ASR 0.5585 -> 0.5550`
- `TPR@1%FPR 0.027 -> 0.021`
- `TPR@0.1%FPR 0.002 -> 0.002`

Read:

1. target-wide `AUC` is slightly better
2. `ASR` is slightly better
3. but low-FPR `TPR@1%FPR` gets materially worse
4. so this still does not beat the original `k16`

## Verdict

- `x113_04_h1_k16_mixture_lambda_down_verdict = negative but useful`

More precise reading:

1. reducing branch frequency does not preserve the original `k16` forgotten-tail benefit
2. it also fails to keep retained low-FPR behavior healthy
3. the current same-family selective-variable candidate is therefore rejected

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `04 current state = original k16 remains the best working instantiation`

Practical read:

1. do not promote `mixture_lambda-down`
2. do not keep iterating same-family scalar tweaks mechanically
3. `04` now needs a broader CPU-side rethink before any new same-family GPU release

## Canonical Evidence Anchor

Primary anchors:

- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-pilot-k16-lambda04375-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-k16-lambda04375-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-retained-k16-lambda04375-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-fullsplit-k16-lambda04375-pairednoise-20260418-r1/summary.json`

Supporting anchors:

- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-k16-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-retained-k16-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-fullsplit-k16-pairednoise-20260418-r1/summary.json`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Platform/Runtime`: no schema change required
- `competition-material sync decision = none`
