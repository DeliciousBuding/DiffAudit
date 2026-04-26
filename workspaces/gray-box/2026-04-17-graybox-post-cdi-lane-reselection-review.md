# 2026-04-17 Gray-Box Post-CDI Lane Reselection Review

## Question

After the paired `CDI` scorer boundary, machine-readable contract, and consumer handoff are all closed, which gray-box lane is now most worth opening next, and does that choice justify any new GPU question?

## Inputs Reviewed

- `D:\Code\DiffAudit\ROADMAP.md`
- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\plan.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-post-temporal-striding-graybox-next-question-review.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-noise-as-probe-challenger-boundary-review.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-noise-as-probe-defended-extension-feasibility-review.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-graybox-sima-feasibility-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-graybox-sima-rescan-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-mofit-current-contract-hold-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-15-graybox-structural-memorization-smoke-note.md`

## Current Gray-Box Truth Before Reselection

Already-closed truths:

- `PIA + stochastic-dropout(all_steps)` remains the admitted defended gray-box headline.
- `TMIA-DM late-window + temporal-striding(stride=2)` remains the strongest defended gray-box challenger reference.
- `CDI` is now a frozen `gray-box collection-level audit extension`, with internal paired scorer boundary and consumer handoff both closed.
- gray-box still has `gpu_release = none`.

This means the next lane should not be:

- more `CDI` scorer interpretation,
- more `TMIA-DM` rung inflation,
- or a mechanical reopen of a branch that already ended in `hold` / `negative but useful`.

## Candidate Review

### 1. `TMIA-DM` packaging follow-up

Not selected.

Reason:

- the branch is already decision-grade for the current question;
- the defended challenger wording is already synchronized;
- reopening it now would be packaging churn, not a new bounded research question.

### 2. `Noise as a Probe`

Selected as the next live lane.

Reason:

- it is the strongest unpromoted genuinely new gray-box mechanism still below the packaged hierarchy;
- it already has repeat-positive bounded local evidence, so the open question is no longer "can it run" but "what exactly blocks promotion";
- its challenger-boundary note already says the honest next step is `comparison/promotion review or defended extension`;
- defended extension is already frozen as `no-go for now`, which leaves a clear CPU-side next question:
  - can the branch define one honest promotion/comparability path, or should it stay permanently below active challenger status on the current contract?

This is higher value than reopening weaker branches because it can still change project-level method-diversity wording without burning GPU.

### 3. `SimA reopen`

Not selected.

Reason:

- `SimA` is already `execution-feasible but weak`;
- the later-timestep rescan improved the best local rung only to `AUC = 0.584961`, still far below challenger quality;
- current reopen rules already require a fresh paper-faithful hypothesis, and this review round did not uncover one strong enough to outrank `Noise as a Probe`.

### 4. `MoFit reopen`

Not selected.

Reason:

- current verdict is already `execution-positive but signal-weak / current-contract hold`;
- the branch should only reopen after a material contract change, not because `CDI` just closed.

### 5. `structural memorization reopen`

Not selected.

Reason:

- the current local faithful approximation is still direction-negative;
- no new bounded hypothesis surfaced in this round.

## Selection

- `selected_next_live_lane = Noise as a Probe promotion-gap review`

## Immediate Task Shape

The next live task should stay:

- `CPU-only`
- `review-first`
- `non-run`

It should answer:

1. what exact comparability gap keeps `Noise as a Probe` below active challenger status;
2. whether any honest promotion path exists on the current local latent-diffusion contract;
3. if not, what minimal contract shift would be required before the family is worth reopening for execution rather than packaging.

## Verdict

- `graybox_post_cdi_reselection_verdict = positive`
- gray-box still has `no-new-gpu-question`
- the most valuable next CPU-first lane is now:
  - `Noise as a Probe promotion-gap review`
- current rejected alternatives:
  - `TMIA-DM packaging reopen`
  - `SimA reopen`
  - `MoFit reopen`
  - `structural memorization reopen`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `Platform/Runtime`: no immediate handoff
- `competition materials`: no immediate sync

Reason:

- this round changes gray-box task selection truth, but does not yet change higher-layer packaging.
