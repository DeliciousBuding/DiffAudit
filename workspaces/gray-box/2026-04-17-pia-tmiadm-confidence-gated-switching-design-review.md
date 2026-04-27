# 2026-04-17 PIA Vs TMIA-DM Confidence-Gated Switching Design Review

## Question

Given that `PIA vs TMIA-DM` same-split combination is already `positive but bounded`, what is the smallest honest next design for a confidence-gated switching rule, and should it become the next CPU-first offline packet?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-pia-vs-tmiadm-disagreement-exploitation-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-pia-vs-tmiadm-long-window-comparison.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-17-ranking-sensitive-variable-search-review.md`

## Review

### 1. The next step should not be another fixed fusion baseline

The repo already knows:

- `z-score sum` is bounded-positive
- `z-score max` is weaker
- opening GPU from this alone is not justified

So another static fusion rerun would add little.

### 2. The design should test a real gating variable, not hidden supervision

The smallest honest switching rule should use only attack-side scores available at inference time.

That rules out:

- label-aware gating
- class-conditional logic without a real class-side contract
- large learned fusion stacks

### 3. The cleanest next rule is margin-aware switching on normalized scores

Freeze the next bounded design as:

- compute normalized `PIA` and `TMIA-DM` scores on the aligned packet
- derive:
  - dominant-method identity
  - absolute margin gap between the two normalized scores
- test one confidence-gated rule family:
  - if one method exceeds the other by at least a frozen margin threshold, trust the dominant method
  - otherwise fall back to the bounded `z-score sum`

This is better than a pure sum-only rerun because it directly tests whether disagreement frontier cases and high-margin cases should be handled differently.

## Verdict

- `pia_tmiadm_confidence_gated_switching_design_verdict = positive but bounded`
- the next honest CPU-first lane should be:
  - `PIA vs TMIA-DM confidence-gated switching offline packet`
- keep:
  - `gpu_release = none`
  - `next_gpu_candidate = none`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/implementation/challenger-queue.md`: update required
- `Platform/Runtime`: no handoff yet
- `Leader/materials`: no sync yet
