# 2026-04-18 X-110 04 H1 Post-Alpha Parameter Selection Review

## Question

After `X-109` ruled out `k16 + alpha-up` as an improvement, what is the next honest parameter-selection posture inside `04-H1`?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-18-x108-04-h1-k16-next-followup-selection.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-18-x109-04-h1-k16-alpha-up-followup-review.md`
- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-pilot-k16-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-pilot-k16-alpha075-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-review-k16-pairednoise-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-review-k16-alpha075-pairednoise-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-review-retained-k16-pairednoise-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-review-retained-k16-alpha075-pairednoise-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-review-fullsplit-k16-pairednoise-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-review-fullsplit-k16-alpha075-pairednoise-20260418-r1\summary.json`

## What Was Resolved

### 1. Stronger forget pressure is no longer an open question

`X-109` already answered the most direct pressure hypothesis:

1. keep `k16`
2. keep `mixture_lambda = 0.5`
3. keep `32` steps
4. increase `alpha` only

That move failed.

So the current open space is no longer “should we push harder on forget loss?”

### 2. Same-family continuation now needs selectivity, not pressure

The `alpha-up` read is asymmetric:

1. forgotten-subset tails do not improve beyond old `k16`
2. retained companion regresses
3. full-split does not improve enough to count as better

That means the remaining plausible same-family moves, if any, must reduce global exposure rather than increase pressure.

## Parameter Selection Read

### Ruled out

Do not prioritize:

1. another `alpha` increase
2. another immediate `k` sweep
3. another same-family GPU rerun without a new CPU-side selection argument

### Still plausible but not yet released

If `04-H1` continues inside the same family, only two single-variable classes still look honest:

1. **frequency/selectivity variables**
   - example: `mixture_lambda` lower than `0.5`
   - intended effect: keep the current `k16` geometry but expose the model to forget-branch updates less often
2. **budget variables**
   - example: shorter step budget than `32`
   - intended effect: reduce retained/full-split drag while testing whether forgotten-tail lift can stay alive

### Control Read

Neither of those should be released immediately as a new GPU task yet.

The current honest state is:

1. original `k16` remains the best working instantiation
2. `alpha-up` is a completed negative
3. the next move must first be a CPU-side parameter review, not another immediate run

## Verdict

- `x110_04_h1_post_alpha_parameter_selection_verdict = no immediate same-family gpu release`

More precise reading:

1. `04-H1` is still alive
2. but it is no longer honest to treat “increase pressure” as an open near-term lever
3. any same-family continuation must now justify one more selective variable, or else stop

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `04 cpu-side next move = selective-variable review only`

## Canonical Evidence Anchor

- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-18-x110-04-h1-post-alpha-parameter-selection-review.md`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Research/docs/comprehensive-progress.md`: update required
- `Research/docs/mainline-narrative.md`: update required
- `Platform/Runtime`: no schema change required
