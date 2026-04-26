# X-86: Non-Graybox Next-Lane Reselection After X-85 Admitted-Summary Sync

> Date: 2026-04-18
> Owner: Researcher
> Context: X-85 completed admitted-summary quality/cost read-path hardening

## Question

After X-85 admitted-summary sync, what is the next honest non-graybox lane?

## Current Control-Plane State

- `active_gpu_question = none`
- `next_gpu_candidate = G1-A larger shared-surface tri-score rerun review pending separate bounded GPU review`
- G1-A bounded GPU review completed: decision = `hold-review-only` (CPU-bound evaluation, not GPU question)
- X-90 blocked: TMIA-DM 512-sample gap (PIA has 512-sample runs, TMIA-DM only 256-sample)
- I-A higher-layer boundary maintenance: epsilon-trajectory consistency mechanism wording added to leader summary

## Candidate Pool Review

### Black-box
- `recon` mainline: stable, no new hypothesis
- `CLiD`: boundary收口 to evaluator-near local corroboration
- `semantic-auxiliary-classifier`: real challenger, no immediate follow-up
- Paper-backed next-family scouting (X-60/X-61): closed negative (face-image LDM overlaps with semantic-auxiliary-classifier/CDI)

### Gray-box
- `PIA` mainline: stable
- `G1-A tri-evidence audit scorer`: X-89 CPU gate positive, X-90 blocked on TMIA-DM 512-sample gap
- `TMIA-DM + temporal-striding`: defended challenger stable
- `CDI` internal paired scorer: stable

### White-box
- `GSA` mainline: stable
- `W-1 = DPDM`: defended comparator stable
- `GSA2`: bounded secondary corroboration stable
- `WB-CH-4 loss-feature challenger` (X-70-X-78): bounded auxiliary evidence, frozen below immediate continuation
- `DP-LoRA / SMP-LoRA`: metric-split bounded exploration branch, no new GPU question

### Innovation Tracks
- `I-A`: higher-layer boundary maintenance completed (epsilon-trajectory consistency wording added)
- `I-B`: actual bounded falsifier, no honest bounded successor lane (X-54/X-65/X-66)
- `I-C`: translated-contract-only + negative falsifier, no honest bounded successor lane (X-56)
- `I-D`: bounded packet landed + negative actual runner-level rerun, no honest bounded successor lane (X-36)

### Cross-box / System-Consumable
- X-85 admitted-summary sync: completed (Evidence Level + Quality/Cost now exposed)
- Unified attack-defense table: stable
- Challenger queue: stable

## Reselection Logic

1. **G1-A (gray-box tri-evidence audit scorer)**: X-89 CPU gate positive, but X-90 blocked on TMIA-DM 512-sample gap
   - Resolution options:
     - Option A: Run TMIA-DM 512-sample (2-4h GPU) to unblock X-90
     - Option B: Downgrade X-90 to 256-sample rerun (weakens story delta)
     - Option C: Defer X-90 until TMIA-DM 512 assets exist
   - Current honest move: Document blocker, defer GPU decision until asset gap resolved

2. **I-A higher-layer boundary maintenance**: Just completed (epsilon-trajectory consistency wording added to leader summary)

3. **Blocked/hold branches**: No honest reopen
   - `I-B / I-C / I-D`: All reviewed for successor lanes (X-54/X-56/X-36), all negative
   - `XB-CH-2`: needs-assets
   - `Finding NeMo`: actual bounded falsifier, same-family rescue forbidden
   - `LiRA / Strong LiRA`: above bounded host-fit budget

4. **Stale-entry sync**: No visible residue after X-85

5. **Candidate-surface expansion**: Already exhausted in X-60/X-61 (black-box), X-65/X-66 (I-B), X-40-X-46 (I-C fresh hypothesis)

## Verdict

**Next honest lane: X-87 = G1-A blocker resolution review**

**Rationale**:
- G1-A is the current `next_gpu_candidate` per ROADMAP §4.5
- X-89 CPU gate passed honestly with positive canary results
- X-90 blocked on TMIA-DM 512-sample gap (documented in `workspaces/2026-04-18-x90-blocker-tmiadm-512-sample-gap.md`)
- G1-A bounded GPU review (hold-review-only) recommends X-90 as CPU-first bounded review, not GPU fire
- Current honest move: Review blocker resolution options and determine next step

**X-87 scope**:
1. Review X-90 blocker resolution options (run TMIA-DM 512 / downgrade / defer)
2. If TMIA-DM 512 is chosen: Create run scripts, execute GPU runs, retry X-90
3. If downgrade/defer is chosen: Update G1-A status, redirect to other lanes
4. Update Research ROADMAP with X-87 verdict

**Alternative if X-87 defers G1-A**:
- Return to I-A higher-layer boundary maintenance (CPU sidecar)
- Or expand to next candidate surface (but X-60/X-61/X-65/X-66/X-40-X-46 already exhausted)

## Next Step

Execute X-87: G1-A blocker resolution review.
