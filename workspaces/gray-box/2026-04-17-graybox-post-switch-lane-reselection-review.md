# 2026-04-17 Gray-Box Post-Switch Lane Reselection Review

## Question

After the first real `PIA vs TMIA-DM confidence-gated switching` offline packet closed as `negative but useful`, should gray-box still keep the next live CPU-first slot, or should that slot move elsewhere?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-17-pia-tmiadm-confidence-gated-switching-offline-packet.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-17-pia-tmiadm-confidence-gated-switching-design-review.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-17-ranking-sensitive-variable-search-review.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\plan.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\challenger-queue.md`

## Review

### 1. Gray-box has answered the current ranking-sensitive question honestly

The repo now knows:

- `PIA vs TMIA-DM` disagreement is real and bounded-positive
- a first confidence-gated switching rule can be evaluated honestly on aligned packets
- but that first switching rule does not beat `z-score sum` on the undefended surfaces
- and it degrades further on the defended surface

So the current question is no longer "is there a bounded next packet?".

That question has already been answered.

### 2. The next gray-box step is no longer obviously higher-value than competing lanes

Current gray-box state is already rich:

- admitted mainline: `PIA + stochastic-dropout(all_steps)`
- strongest defended challenger reference: `TMIA-DM + temporal-striding(stride=2)`
- strengthened bounded challenger candidate: `Noise as a Probe`
- internal paired extension: `CDI`
- bounded ranking-sensitive packet: `PIA vs TMIA-DM confidence-gated switching`

This means gray-box is no longer under-explored relative to the rest of the repo.

### 3. Reopening gray-box immediately would likely become same-family packet inflation

The most obvious immediate follow-ups would be:

- more margin-threshold scans
- alternate confidence heuristics on the same surfaces
- another bounded switch variant without a new mechanism

Those would currently look like packet inflation rather than a stronger mainline move.

## Verdict

- `graybox_post_switch_lane_reselection_verdict = positive`

More precise reading:

1. gray-box does **not** currently expose a more urgent immediate CPU-first lane than the best remaining cross-box or other-box questions;
2. the just-finished switching packet should therefore be treated as a clean closure point for the current gray-box ranking-sensitive branch;
3. the next live CPU-first slot should yield away from gray-box for now.

## Carry-Forward Rule

- keep:
  - `gpu_release = none`
  - `next_gpu_candidate = none`
- keep the current gray-box package stable:
  - `PIA + stochastic-dropout(all_steps)` headline
  - `TMIA-DM + temporal-striding(stride=2)` strongest defended challenger reference
  - `Noise as a Probe` strengthened bounded challenger candidate
  - `CDI` internal paired extension
  - `PIA vs TMIA-DM confidence-gated switching` as a bounded analysis packet
- do not reopen the switching branch unless a genuinely new gating signal appears

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `workspaces/implementation/challenger-queue.md`: update required
- `Leader/materials`: no immediate sync
- `Platform/Runtime`: no direct handoff
