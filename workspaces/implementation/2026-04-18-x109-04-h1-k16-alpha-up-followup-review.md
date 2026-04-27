# 2026-04-18 X-109 04 H1 K16 Alpha-Up Follow-Up Review

## Question

After `X-108` selected `k16 + alpha-up` as the next bounded single-variable follow-up, does increasing the forget pressure from `alpha = 0.5` to `alpha = 0.75` improve the current `04-H1` working instantiation?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-18-x108-04-h1-k16-next-followup-selection.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-pilot-k16-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-pilot-k16-alpha075-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-k16-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-k16-alpha075-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-retained-k16-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-retained-k16-alpha075-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-fullsplit-k16-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-fullsplit-k16-alpha075-pairednoise-20260418-r1/summary.json`

## What Landed

### 1. One real `k16 alpha-up` pilot now exists

The repo now contains one actual changed pilot with:

- `k = 16`
- `alpha = 0.75`
- `mixture_lambda = 0.5`
- `32` steps
- same target checkpoint and optimizer contract as the previous `k16`

### 2. One full paired-noise tri-board also exists for the alpha-up rung

The new rung now has:

1. forgotten subset board
2. retained high-risk companion board
3. full-split board

So the parameter change can be judged directly against the old `k16` baseline on the same review surface.

## Actual Read

### Pilot-side training read

Original `k16`:

- `mean_keep_loss = 0.024214`
- `mean_forget_loss = 0.022605`
- `mean_objective = 0.019623`

`k16 alpha-up`:

- `mean_keep_loss = 0.024331`
- `mean_forget_loss = 0.022766`
- `mean_objective = 0.017394`

Read:

1. the branch schedule is unchanged (`19 / 13`)
2. the stronger `alpha` mainly pushes the objective lower on forget-branch steps
3. this is a real pressure increase, not a logging artifact

### Forgotten subset

Old `k16`:

- `AUC 0.903509 -> 0.885965`
- `TPR@1%FPR 0.315789 -> 0.368421`
- `TPR@0.1%FPR 0.315789 -> 0.368421`

`k16 alpha-up`:

- `AUC 0.903509 -> 0.883041`
- `TPR@1%FPR 0.315789 -> 0.368421`
- `TPR@0.1%FPR 0.315789 -> 0.368421`

Read:

1. the useful tail gain does not improve further
2. `AUC` gets slightly worse
3. so the extra pressure does not buy a better forgotten-subset outcome

### Retained companion

Old `k16`:

- `AUC 0.781046 -> 0.781046`
- `TPR@1%FPR 0.235294 -> 0.294118`
- `TPR@0.1%FPR 0.235294 -> 0.294118`

`k16 alpha-up`:

- `AUC 0.781046 -> 0.774510`
- `TPR@1%FPR 0.235294 -> 0.235294`
- `TPR@0.1%FPR 0.235294 -> 0.235294`

Read:

1. the old retained-tail improvement disappears
2. `AUC` now also drops
3. this is a clear regression relative to the current working instantiation

### Full split

Old `k16`:

- `AUC 0.623331 -> 0.622141`
- `ASR 0.5585 -> 0.5675`
- `TPR@1%FPR 0.027 -> 0.026`
- `TPR@0.1%FPR 0.002 -> 0.002`

`k16 alpha-up`:

- `AUC 0.623331 -> 0.622931`
- `ASR 0.5585 -> 0.5690`
- `TPR@1%FPR 0.027 -> 0.024`
- `TPR@0.1%FPR 0.002 -> 0.002`

Read:

1. `AUC` is fractionally closer to neutral
2. but low-FPR `TPR@1%FPR` is worse
3. `ASR` drift is slightly larger
4. this does not count as a real improvement

## Verdict

- `x109_04_h1_k16_alpha_up_verdict = negative but useful`

More precise reading:

1. stronger forget pressure does not improve the forgotten-subset tails beyond the old `k16`
2. it clearly hurts the retained companion board
3. it does not improve the target-wide full-split read enough to justify promotion

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `04 current state = original k16 remains the best working instantiation`

Practical read:

1. do not promote `alpha-up`
2. do not replace the original `k16`
3. the next `04-H1` move now needs a fresh CPU-side parameter-selection review rather than another immediate GPU rerun

## Canonical Evidence Anchor

Primary anchors:

- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-pilot-k16-alpha075-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-k16-alpha075-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-retained-k16-alpha075-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-fullsplit-k16-alpha075-pairednoise-20260418-r1/summary.json`

Supporting anchors:

- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-k16-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-retained-k16-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-fullsplit-k16-pairednoise-20260418-r1/summary.json`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Platform/Runtime`: no schema change required
- `competition-material sync decision = none`
