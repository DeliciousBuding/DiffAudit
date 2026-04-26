# 2026-04-16 Black-Box Score-Package Contract Selection Verdict

## Task

- select the next bounded contract surface for `BB-6` after the initial alignment blocker verdict

## Question

- If `Recon` and `semantic-auxiliary-classifier` cannot currently be packaged on their frozen artifacts, which single `Recon` contract surface is the shortest honest target for the first aligned semantic-aux comparator?

## Evidence Base

- `workspaces/black-box/2026-04-10-recon-decision-package.md`
- `workspaces/black-box/plan.md`
- `experiments/recon-runtime-mainline-ddim-public-50-step10/summary.json`
- `experiments/recon-runtime-mainline-ddim-public-100-step30/summary.json`

## Candidates Considered

### `public-100 step30`

Pros:

- current `main evidence`
- most complete and most defensible `Recon` artifact chain

Cons:

- larger sample budget
- tied to the headline wording layer, so early semantic-aux alignment noise would directly contaminate the strongest black-box package

### `public-50 step10`

Pros:

- already frozen as `best single metric reference`
- explicit target/shadow score-artifact structure already exists
- uses the same admitted `celeba_partial_target/checkpoint-25000` contract surface
- smaller and therefore cheaper first alignment target

Cons:

- not the headline `main evidence`

## Verdict

- `positive selection`

The first aligned semantic-aux comparator should target:

- `Recon DDIM public-50 step10`

## Decision

Current decision:

- use `public-50 step10` as the first `BB-6.2` contract surface
- keep `public-100 step30` untouched as the current headline package while the new comparator is still exploratory
- keep `gpu_release = bounded only` for the first aligned comparator; no broad rerun ladder yet

## Why This Is Best

1. it is the shortest honest bridge from current `semantic-aux` artifacts to admitted `Recon` semantics;
2. it preserves the stronger `public-100 step30` package from premature contamination;
3. it already exposes target/shadow score artifacts and the exact `celeba_partial_target` checkpoint surface that the aligned comparator should match.

## Handoff Note

- `Platform`: no sync needed.
- `Runtime`: no sync needed.
- `Materials`: do not change headline black-box wording yet; this is only the selected alignment target for the next bounded comparator.
