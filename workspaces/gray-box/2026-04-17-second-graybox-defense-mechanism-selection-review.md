# 2026-04-17 Second Gray-Box Defense Mechanism Selection Review

## Question

After black-box candidate refresh closed negatively and gray-box regained the near-term innovation slot, which existing gray-box defense mechanism should now be frozen as the honest `second gray-box defense mechanism`, and does that selection justify any new GPU release?

## Inputs Reviewed

- `D:\Code\DiffAudit\ROADMAP.md`
- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\README.md`
- `D:\Code\DiffAudit\Research\docs\comprehensive-progress.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\README.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\plan.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-tmiadm-temporal-striding-defense-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-pia-vs-tmiadm-temporal-striding-defended-comparison.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-post-temporal-striding-graybox-next-question-review.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-mofit-current-contract-hold-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-15-graybox-defense-precision-throttling-note.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-15-graybox-epsilon-output-noise-defense-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\challenger-queue.md`

## Review

### 1. The second gray-box defense question is now about selection, not exploration breadth

Current gray-box defense truth is already narrow enough:

- `PIA + stochastic-dropout(all_steps)` remains the admitted defended headline
- `TMIA-DM late-window + temporal-striding(stride=2)` is the strongest defended challenger-specific branch
- `Noise as a Probe` has no honest defended-extension gate on the current local contract
- `MoFit` is execution-positive but signal-weak under the current contract

So the right question is no longer "what can we still try cheaply." It is "which mechanism is strong enough and different enough to deserve freezing as the second gray-box defense mechanism."

### 2. `TMIA-DM temporal-striding(stride=2)` is the only mechanism that still survives honest selection pressure

Why it wins:

1. it is materially different from the admitted `stochastic-dropout(all_steps)` story
   - it weakens a challenger by restricting time-resolution exposure rather than injecting stochastic instability
2. it already survived bounded CPU review plus `GPU128` and `GPU256`
3. the comparison layer already shows it is stronger than `TMIA + dropout` on the same challenger family
4. it changes defended gray-box ordering without pretending to replace the admitted `PIA` headline

That makes it the only currently honest candidate for a frozen second defense mechanism.

### 3. The other visible candidates fail the selection test for different reasons

Rejected or non-promotable options:

1. `epsilon-precision-throttling`
   - bounded smoke already closed negative
   - it increased attack quality instead of suppressing it

2. `epsilon-output-noise`
   - bounded smoke also closed negative
   - it raised cost without improving the defense outcome

3. `input-gaussian-blur`
   - current roadmap/plan truth already records it as direction-negative
   - it is not a credible next defense mechanism under the current contract

4. `Noise as a Probe` defense follow-up
   - current contract does not expose an honest defended-extension gate
   - reopening it would be contract drift, not mechanism selection

5. `MoFit`
   - current contract is execution-positive but signal-weak
   - this is a hold candidate, not a second defended mechanism

## Verdict

- `second_graybox_defense_mechanism_selection_verdict = positive but bounded`
- freeze:
  - `PIA + stochastic-dropout(all_steps)` as admitted defended headline
  - `TMIA-DM late-window + temporal-striding(stride=2)` as the selected second gray-box defense mechanism
- do **not** treat that as a project-wide replacement defense
- do **not** open a new GPU task from this selection by itself
- current global posture remains:
  - `gpu_release = none`
  - `next_gpu_candidate = none`
- the next live CPU-first lane should now move to:
  - `distinct white-box defended-family import / selection`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `workspaces/implementation/challenger-queue.md`: update required
- `docs/comprehensive-progress.md`: light sync required
- `README.md`: light sync suggested
- `Platform/Runtime`: no direct handoff required
- `Leader/materials`: suggestion-only; gray-box can now honestly be summarized as `headline defended pair + selected second mechanism`, but the second mechanism remains challenger-specific
