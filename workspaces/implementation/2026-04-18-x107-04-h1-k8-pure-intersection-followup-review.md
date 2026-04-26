# 2026-04-18 X-107 04 H1 K8 Pure-Intersection Follow-Up Review

## Question

After `X-106` established `k16` as the current best working instantiation, does the first pure-intersection lower-bound pilot (`k8`) produce a cleaner and more honest `04-H1` lead candidate?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-prep-full-overlap-k8-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-pilot-k8-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-retained-companion-k8-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-review-k8-pairednoise-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-review-retained-k8-pairednoise-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-review-fullsplit-k8-pairednoise-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-review-k16-pairednoise-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-review-retained-k16-pairednoise-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-review-fullsplit-k16-pairednoise-20260418-r1\summary.json`

## What Landed

### 1. One first pure-intersection pilot now exists

`k8` is the first `04-H1` rung that does not fall back to aggregate-percentile ranking.

Its forget set is now constrained to the exact high-risk overlap surface:

1. `Top10%(GSA)`
2. `Top10%(PIA)`
3. exact shared member overlap only

So this is the cleanest lower-bound probe the current overlap geometry can honestly support.

### 2. One full paired-noise tri-board also exists for `k8`

`k8` now has:

1. forgotten subset board
2. retained high-risk companion board
3. full-split board

This means the pure-overlap read can be compared directly against `k16` on the same paired-noise review surface.

## Actual Read

### `k8` paired-noise target-transfer metrics

Forgotten subset:

- baseline:
  - `AUC = 0.901235`
  - `ASR = 0.500000`
  - `TPR@1%FPR = 0.444444`
  - `TPR@0.1%FPR = 0.444444`
- defended:
  - `AUC = 0.901235`
  - `ASR = 0.500000`
  - `TPR@1%FPR = 0.444444`
  - `TPR@0.1%FPR = 0.444444`

Retained companion:

- baseline:
  - `AUC = 0.922222`
  - `ASR = 0.526316`
  - `TPR@1%FPR = 0.800000`
  - `TPR@0.1%FPR = 0.800000`
- defended:
  - `AUC = 0.911111`
  - `ASR = 0.526316`
  - `TPR@1%FPR = 0.800000`
  - `TPR@0.1%FPR = 0.800000`

Full split:

- baseline:
  - `AUC = 0.623331`
  - `ASR = 0.558500`
  - `TPR@1%FPR = 0.027000`
  - `TPR@0.1%FPR = 0.002000`
- defended:
  - `AUC = 0.621005`
  - `ASR = 0.554500`
  - `TPR@1%FPR = 0.025000`
  - `TPR@0.1%FPR = 0.002000`

### Comparison to `k16`

Relative to `k16` paired-noise:

1. forgotten subset:
   - `k16` still showed useful low-FPR tail lift despite a small `AUC` loss
   - `k8` becomes almost exact neutrality instead of a targeted gain
2. retained companion:
   - `k16` preserved tail improvement on the retained high-risk board
   - `k8` no longer improves either low-FPR tail and also loses `AUC`
3. full split:
   - `k8` is slightly cleaner than `k16`
   - but the improvement is too small to justify giving up the still-useful `k16` forgotten/retained behavior

### Shift read

On the paired-noise full-split board for `k8`:

- all members mean loss shift = `+0.000166`
- forgotten members mean loss shift = `+0.001872`
- all nonmembers mean loss shift = `-0.000187`

This is the cleanest drift profile seen so far inside `04-H1`.

But that cleanliness now comes with a stronger over-tightening signal:

1. target-wide drift is nearly neutral
2. forgotten-only movement is still present, but much smaller than what would justify promotion
3. the retained board no longer preserves the small tail benefit that kept `k16` alive

## Verdict

- `x107_04_h1_k8_pure_intersection_verdict = cleaner but too weak`

More precise reading:

1. `k8` proves the pure-overlap route is executable
2. it is useful as a lower-bound cleanliness probe
3. it does not replace `k16` as the current working instantiation

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = one more bounded follow-up around the k16 regime only`
- `04 current state = k16 remains lead; k8 archived as lower-bound cleanliness probe`

Practical read:

1. do not tighten further just to chase cleaner drift
2. do not reinterpret `k8` neutrality as defense-positive
3. keep `k16` as the honest lead inside `04-H1`

## Canonical Evidence Anchor

Primary anchors:

- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-pilot-k8-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-review-k8-pairednoise-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-review-retained-k8-pairednoise-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-review-fullsplit-k8-pairednoise-20260418-r1\summary.json`

Supporting anchors:

- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-review-k16-pairednoise-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-review-retained-k16-pairednoise-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-review-fullsplit-k16-pairednoise-20260418-r1\summary.json`

## Handoff Decision

- `Research/ROADMAP.md`: already synced
- `Research/docs/comprehensive-progress.md`: already synced
- `Research/docs/mainline-narrative.md`: already synced
- `Research/docs/future-phase-e-intake.md`: already synced
- `Platform/Runtime`: research-side note only; no new API contract required
