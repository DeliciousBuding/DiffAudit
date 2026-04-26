# 2026-04-16 Cross-box Transfer/Portability Review

## Task

- `X-4.3` transfer or portability probe when assets permit

## Question

- After the latest cross-box agreement and fusion reviews, is there now an honest cross-box transfer/portability probe that can be started, or does this branch still remain asset-blocked?

## Inputs Reviewed

- `workspaces/implementation/2026-04-16-crossbox-agreement-analysis-refresh.md`
- `workspaces/implementation/2026-04-16-crossbox-score-calibration-review.md`
- `workspaces/implementation/challenger-queue.md`
- `docs/comprehensive-progress.md`
- current `black-box / gray-box / white-box` workspace plans

## Candidate Transfer Questions

Current plausible transfer/portability questions are:

1. cross-dataset transfer
   - train on one dataset, test on another
2. cross-model transfer
   - train on `DDPM`, test on `DDIM` or vice versa
3. cross-threat-model agreement on overlapping assets
   - score-level portability between boxes

## Current Gate Check

### A. Cross-dataset transfer

Still not ready.

Reason:

- no paired source/target models on different datasets are frozen as a current branch packet

### B. Cross-model transfer

Still not ready.

Reason:

- no paired `DDPM/DDIM` models on the same dataset and same split contract are frozen for a bounded probe

### C. Cross-threat-model score portability

Still not ready as an execution item.

Reason:

- the three boxes currently play different project roles
- score semantics are not aligned enough for a portable transfer claim
- `X-4.2` already closed the scalar-fusion path as dishonest at current repo state

## Decision

So `X-4.3` remains:

- `needs-assets`

and should not be promoted into an active run.

The honest next move is not a portability probe.

It is:

- new-family candidate generation
- or within-box work that creates the missing paired assets

## Verdict

- `negative but useful`

Current conclusion:

- no current cross-box transfer/portability probe is honest to start
- this branch remains asset-blocked
- no new GPU question is justified by this review

## Handoff Decision

- `Leader / materials`: no sync needed
- `Platform`: no sync needed
- `Runtime`: no sync needed
- `GPU`: no new GPU admission question is justified by this analysis

## Next Recommendation

1. keep `X-4.3` closed as `needs-assets`
2. if this branch is reopened later, require:
   - paired model contracts
   - paired split contracts
   - one bounded portability hypothesis
