# 2026-04-16 Cross-box Score Calibration Review

## Task

- `X-4.2` score calibration or fusion with bounded hypothesis

## Question

- After the latest cross-box agreement refresh, is it now honest to build one calibrated or fused cross-box score, or should the project continue to resist collapsing the three boxes into a single scalar ranking?

## Inputs Reviewed

- `docs/admitted-results-summary.md`
- `docs/leader-research-ready-summary.md`
- `docs/comprehensive-progress.md`
- `workspaces/implementation/2026-04-15-attack-defense-matrix.md`
- `workspaces/implementation/2026-04-16-crossbox-agreement-analysis.md`
- `workspaces/implementation/2026-04-16-crossbox-agreement-analysis-refresh.md`
- current `black-box / gray-box / white-box` workspace plans

## Calibration Hypothesis

Candidate idea:

- build one normalized score or fused ranking across:
  - black-box
  - gray-box
  - white-box

Intended benefit:

- simplify project-level communication
- make it easier to compare lines on one scale

## Why It Is Not Honest Yet

The current repo state still does **not** support one cross-box scalar.

Reasons:

1. access levels differ materially
   - `black-box` asks “can risk be seen at weakest access?”
   - `gray-box` asks “what is the best attack-defense narrative?”
   - `white-box` asks “what is the upper-bound / deepest contrast?”
2. metric meaning differs by role
   - some rows are best used as risk proofs
   - some as defended baselines
   - some as bridge or depth evidence
3. boundary conditions differ
   - admitted lines have different provenance and external-validity caveats
   - challenger lines move faster than headline lines
4. any scalar fusion now would hide the exact thing the current project narrative needs to preserve:
   - box-specific role split

## What Is Honest Instead

The honest current structure is:

- one layered project narrative
- plus per-box headline / challenger / boundary fields

This means:

- keep `black-box = existence proof`
- keep `gray-box = main attack-defense story`
- keep `white-box = depth / upper bound`

If a future system wants more comparability, the honest next step is not scalar fusion first.

It is:

- role-aware presentation
- or bounded within-role calibration

not:

- one unified score across all three boxes

## Verdict

- `negative but useful`

Current conclusion:

- cross-box score calibration/fusion should **not** be promoted right now
- the repo should keep a layered role-based summary instead of a scalar fusion
- no new GPU question is justified by this review

## Handoff Decision

- `Leader / materials`: yes, wording-only
  - avoid any “overall best score” framing
  - preserve role-specific descriptions
- `Platform`: optional future enhancement only
  - if UI needs synthesis, expose `role` and `boundary` fields rather than a single fused score
- `Runtime`: no sync needed
- `GPU`: no new GPU admission question is justified by this analysis

## Next Recommendation

1. keep `X-4.2` closed as a current no-go for scalar fusion
2. if cross-box work continues, prefer:
   - within-role calibration
   - transfer/portability questions
   - new-family candidate generation
