# X-88: Non-Graybox Next-Lane Reselection After X-87 G1-A Defer

> Date: 2026-04-18
> Owner: Researcher
> Context: X-87 deferred G1-A/X-90 (TMIA-DM 512-sample gap conflicts with CPU-first classification)

## Question

After X-87 deferred G1-A, what is the next honest non-graybox lane?

## Current Control-Plane State

- `active_gpu_question = none`
- `next_gpu_candidate = none` (G1-A deferred to needs-assets)
- G1-A status: `hold-review-only` → `deferred-needs-assets` (TMIA-DM 512-sample gap)
- X-90 status: `blocked` → `deferred` (asset gap, conflicts with CPU-first classification)
- I-A higher-layer boundary maintenance: completed (epsilon-trajectory consistency wording added)

## Candidate Pool Review

### Root ROADMAP B-M0 Context (2026-04-17 override)
- 48h bounded GPU release candidate window
- Candidate A (default): bounded shadow-LR / likelihood-ratio follow-up on frozen admitted white-box loss-score packet
- Candidate B (story-changing backup): I-C in-model white-gray bridge on frozen matched pair
- Kill gate: 24h CPU-first scoping must give non-ambiguous choice

### Black-box
- No new hypothesis, all recent scoping closed negative (X-60/X-61)

### Gray-box
- G1-A: deferred-needs-assets
- PIA mainline: stable
- TMIA-DM + temporal-striding: stable

### White-box
- GSA mainline: stable
- WB-CH-4 loss-feature (X-70-X-78): bounded auxiliary evidence, frozen below immediate continuation
- Root ROADMAP B-M0 Candidate A: bounded shadow-LR / likelihood-ratio follow-up on frozen admitted white-box loss-score packet

### Innovation Tracks
- I-A: higher-layer boundary maintenance completed
- I-B: actual bounded falsifier, no successor (X-54/X-65/X-66)
- I-C: translated-contract-only + negative falsifier, no successor (X-56)
  - Root ROADMAP B-M0 Candidate B: I-C in-model white-gray bridge on frozen matched pair
- I-D: bounded packet + negative rerun, no successor (X-36)

## Reselection Logic

**Root ROADMAP B-M0 takes priority** (2026-04-17 chief-designer override):
- Axis B aggressive mode: release 1 bounded GPU candidate
- Current live lane should be B-M0 scoping, not X-N reselection loop

**B-M0 Candidate A (default)**: bounded shadow-LR / likelihood-ratio follow-up on frozen admitted white-box loss-score packet
- Assets: X-75/X-77 frozen admitted `DDPM/CIFAR10` white-box loss-score packet (`extraction_max_samples = 64`)
- Hypothesis: shadow-distribution LR-transfer beats X-77 threshold-transfer on low-FPR targets
- Same packet identity, same schema, same runtime surface (frozen)
- Host budget: CIFAR10 shadow distribution construction fits 4070 Laptop 8GB
- Expected: positive/negative/bounded, all produce admitted-level white-box distinct-scorer auxiliary result

**B-M0 Candidate B (story-changing backup)**: I-C in-model white-gray bridge on frozen matched pair
- Assets: Research §5.3 frozen `1 member + 1 nonmember` matched pair
- Blocker: §5.3 blocker B requires `in-model intervention surface` (not offline tensor masking)
- 24h gate: only if blocker B resolves within 24h, else default to Candidate A

**Current honest reading**:
- B-M0 Candidate A is CPU-first scoping ready (frozen packet, bounded hypothesis, host-fit approved)
- B-M0 Candidate B has 24h blocker (in-model intervention surface)
- Root ROADMAP §4 B-M0 kill gate: 24h CPU-first scoping must give non-ambiguous choice

## Verdict

**Next honest lane: X-88 = B-M0 Candidate A CPU-first gate (bounded shadow-LR white-box loss-score follow-up)**

**Rationale**:
1. Root ROADMAP B-M0 takes priority over X-N reselection loop (2026-04-17 override)
2. Candidate A is CPU-first scoping ready (frozen packet, bounded hypothesis, host-fit approved)
3. Candidate B has 24h blocker (in-model intervention surface not yet implemented)
4. X-75/X-77 frozen white-box loss-score packet provides same-asset identity for LR follow-up
5. Shadow-LR hypothesis is genuinely new (distinct from X-77 threshold-transfer)

**X-88 scope**:
1. Review X-75/X-77 frozen packet contract (extraction_max_samples=64, shadow-threshold-transfer)
2. Define bounded shadow-LR hypothesis (LR-transfer vs threshold-transfer on low-FPR targets)
3. Review host-fit budget (CIFAR10 shadow distribution construction on 4070 Laptop 8GB)
4. Review story delta (white-box distinct-scorer auxiliary evidence)
5. Define kill gate (explicit no-fire conditions)
6. Verdict: ready-for-bounded-gpu-review / hold / no-go

**Kill gate check**: 24h CPU-first scoping window starts now

## Next Step

Execute X-88: B-M0 Candidate A CPU-first gate (bounded shadow-LR white-box loss-score follow-up).
