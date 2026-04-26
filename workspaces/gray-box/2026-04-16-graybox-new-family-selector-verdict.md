# 2026-04-16 Gray-Box New-Family Selector Verdict

## Question

After the current `TMIA-DM` challenger and its defended variants are already packaged, what should become the next live gray-box branch?

## Inputs Reviewed

- `Research/ROADMAP.md`
- `workspaces/gray-box/plan.md`
- `workspaces/white-box/plan.md`
- `workspaces/black-box/plan.md`
- existing `INF-3` backlog-critic output

## Current State

1. Gray-box mainline and defended challenger packaging are already strong enough for the current round.
2. The gray-box plan explicitly says the next shortest step should be:
   - another lane, or
   - a genuinely new gray-box mechanism
3. White-box just consumed a bounded CPU round on `DP-LoRA` dossier hardening.
4. Black-box explicitly remains in `no immediate rerun` posture.

## Selection Review

### Selected

- `selected_next_lane = GB-5 gray-box genuinely-new-family selector`

Why:

1. It restores innovation pressure on the strongest narrative box.
2. It avoids wasting GPU on already-packaged `TMIA-DM` branches.
3. It is the shortest path to the next real gray-box verdict.

### Not Selected Now

- `X-4.2 score calibration/fusion`
  - useful system work, but weaker blocker leverage than opening a new family
- `WB-next import/shortlist`
  - white-box just received a bounded candidate-generation round
- `black-box reopen`
  - current black-box plan still says `no immediate rerun`

## Immediate Task Shape

This new selector lane should stay:

- `CPU-only`
- `candidate-generation only`
- `one-family selection only`

Expected output:

1. choose one next family
2. reject the near-miss alternatives for now
3. define one first bounded smoke
4. define future `gpu_release` conditions

## Verdict

- `selection_verdict = positive`
- `gpu_release = none`
- `admitted_change = none`
- `next_step = open GB-5 selector note and freeze one first-smoke candidate`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this is a research-priority reselection only.
