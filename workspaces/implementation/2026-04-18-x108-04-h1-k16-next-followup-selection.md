# 2026-04-18 X-108 04 H1 K16 Next Follow-Up Selection

## Question

After `X-107` confirmed that `k16` remains the current best working instantiation, what is the next honest bounded follow-up inside `04-H1`?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-prep-full-overlap-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-pilot-k16-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-pilot-k32-20260418-r2/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-pilot-k8-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-k16-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-retained-k16-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-fullsplit-k16-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-k8-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-retained-k8-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-fullsplit-k8-pairednoise-20260418-r1/summary.json`

## What Was Reviewed

### 1. `k` is no longer the open variable

The current three-rung picture is already informative enough:

1. `k32` is too broad and drifts too much
2. `k8` is clean but over-tightened
3. `k16` is the only rung that still preserves some low-FPR tail benefit while keeping full-split near neutral

So the next follow-up should not spend another bounded pilot on moving `k` again.

### 2. The remaining open question is pressure, not geometry

Across `k8 / k16 / k32`, the training contract is still:

- `alpha = 0.5`
- `mixture_lambda = 0.5`
- `32` steps

The geometry change already told us where the forget-set size should sit.

What remains unresolved is whether the current `k16` pressure is slightly too weak:

1. forgotten-subset `AUC` still drops
2. full-split drift is small enough that there is still some headroom
3. retained tails are improved enough that the rung should not be abandoned yet

## Actual Read

### Pilot-side training comparison

`k16` training statistics:

- `branch_counts = keep_only 19 / keep_minus_forget 13`
- `mean_keep_loss = 0.024214`
- `mean_forget_loss = 0.022605`
- `mean_objective = 0.019623`

`k32` training statistics:

- `branch_counts = keep_only 17 / keep_minus_forget 15`
- `mean_keep_loss = 0.026602`
- `mean_forget_loss = 0.022513`
- `mean_objective = 0.021325`

`k8` training statistics:

- `branch_counts = keep_only 20 / keep_minus_forget 12`
- `mean_keep_loss = 0.025106`
- `mean_forget_loss = 0.020584`
- `mean_objective = 0.021246`

Read:

1. the branch mix is already close enough to the intended `0.5` mixture on all three runs
2. the decisive difference is therefore not branch frequency noise
3. the next bounded change should be one single pressure variable, not another `k` move

### Why `alpha` is the best next single variable

The next follow-up should keep:

- `k = 16`
- `mixture_lambda = 0.5`
- `32` steps

and change only:

- `alpha: 0.5 -> 0.75`

Reason:

1. changing `mixture_lambda` would change how often forget batches are sampled, which is more likely to alter global drift frequency
2. changing `alpha` keeps the same forget geometry and same branch schedule, while only strengthening the forget term when that branch is already selected
3. `k16` full-split is currently close enough to neutral that a moderate `alpha` increase is the cleanest next bounded probe

## Verdict

- `x108_04_h1_next_followup_selection = k16 plus alpha-up single-variable pilot`

More precise reading:

1. do not reopen `k32`
2. do not tighten further below `k16`
3. do not switch family yet
4. the next honest GPU candidate is one `k16` changed pilot with `alpha = 0.75`

## Proposed Budget

One bounded pilot only:

- forget set: current exported `k16`
- checkpoint: current target `checkpoint-9600`
- steps: `32`
- `alpha = 0.75`
- `mixture_lambda = 0.5`
- same paired-noise tri-board review after training

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = 04-H1 k16 alpha-up pilot (alpha=0.75, mixture_lambda=0.5, 32 steps)`
- `cpu_sidecar = keep system-consumable Research -> Runtime -> Platform sync aligned`

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-18-x108-04-h1-k16-next-followup-selection.md`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Platform/Runtime`: no schema change required for the next pilot selection itself
- `competition-material sync decision = none`
