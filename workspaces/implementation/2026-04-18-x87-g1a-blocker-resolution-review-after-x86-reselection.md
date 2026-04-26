# X-87: G1-A Blocker Resolution Review After X-86 Reselection

> Date: 2026-04-18
> Owner: Researcher
> Context: X-86 identified G1-A blocker resolution as next honest lane

## Question

How should X-90 TMIA-DM 512-sample blocker be resolved?

## Blocker Summary

**X-90 blocked**: PIA has 512-sample runs, TMIA-DM only has 256-sample runs
- Identity alignment enforces matching sample counts
- Blocker documented: `workspaces/2026-04-18-x90-blocker-tmiadm-512-sample-gap.md`

## Resolution Options Review

### Option A: Run TMIA-DM 512-sample (GPU required)
- **Cost**: 2-4h GPU time (2 surfaces: undefended + defended)
- **Verdict**: Unblocks X-90 honestly
- **Rationale**: G1-A review justified story delta as testing tri-score robustness at scale (512-sample = 2x X-89 canary)
- **Risk**: GPU slot consumption, host health

### Option B: Downgrade X-90 to 256-sample rerun (CPU-only)
- **Cost**: Minimal (X-89 already evaluated 256-sample surfaces)
- **Verdict**: Weakens story delta justification from G1-A review
- **Rationale**: Larger rerun would only verify robustness, not test new scale
- **Risk**: G1-A promotion weakened

### Option C: Defer X-90 until TMIA-DM 512 assets exist
- **Cost**: Delays G1-A promotion
- **Verdict**: Honest but delays
- **Rationale**: Keep G1-A as CPU-bound canary (X-89), wait for assets
- **Risk**: G1-A lane stalls

## Current Control-Plane State

- `active_gpu_question = none`
- `next_gpu_candidate = G1-A larger shared-surface tri-score rerun review pending separate bounded GPU review`
- G1-A bounded GPU review: decision = `hold-review-only` (CPU-bound evaluation, not GPU question)
- Root ROADMAP §4 B-M0: 48h bounded GPU release candidate window (2026-04-17 override)
- Root ROADMAP §6.2: Allow max 1 active GPU question before 2026-04-19

## Decision Logic

**G1-A bounded GPU review recommendation**: X-90 as CPU-first bounded review, not GPU slot

**Root ROADMAP B-M0 context**: 
- Candidate A (default): bounded shadow-LR / likelihood-ratio follow-up on frozen admitted white-box loss-score packet
- Candidate B (story-changing backup): I-C in-model white-gray bridge on frozen matched pair
- Kill gate: 24h CPU-first scoping must give non-ambiguous choice

**Current honest reading**:
- G1-A X-90 is CPU-bound offline evaluation (no GPU inference required)
- TMIA-DM 512-sample gap blocks X-90 execution
- Running TMIA-DM 512 would consume GPU slot (2-4h)
- Root ROADMAP allows 1 GPU question, but B-M0 candidates take priority over G1-A

**Conflict**: G1-A review says "CPU-first bounded review", but TMIA-DM 512 requires GPU

## Verdict

**Decision: Option C (Defer X-90 until TMIA-DM 512 assets exist)**

**Rationale**:
1. G1-A bounded GPU review explicitly classified X-90 as "CPU-bound offline evaluation, not GPU question"
2. TMIA-DM 512-sample run requires GPU (2-4h), conflicts with CPU-first classification
3. Root ROADMAP B-M0 takes priority for GPU slot (bounded white-box/cross-box candidates)
4. X-89 CPU gate already passed honestly with positive 256-sample canary results
5. G1-A can remain as bounded gray-box auxiliary evidence candidate without X-90

**Next steps**:
1. Update G1-A status: `hold-review-only` → `deferred-needs-assets` (TMIA-DM 512-sample gap)
2. Update X-90 status: `blocked` → `deferred` (asset gap, conflicts with CPU-first classification)
3. Redirect to B-M0 CPU-first scoping (bounded white-box/cross-box candidates per root ROADMAP)
4. TMIA-DM 512 can be revisited when GPU slot is available and not blocked by higher-priority candidates

**Kill gate check**: X-90 blocker resolution took <24h CPU-first review, decision is non-ambiguous (defer)

## Alternative if B-M0 also defers

If B-M0 CPU-first scoping also results in no GPU release:
- Return to I-A higher-layer boundary maintenance (CPU sidecar)
- Or expand to next candidate surface
- Keep `active_gpu_question = none` until genuinely new bounded hypothesis appears

## Next Step

Execute X-88: Update G1-A/X-90 status to deferred, redirect to B-M0 CPU-first scoping per root ROADMAP.
