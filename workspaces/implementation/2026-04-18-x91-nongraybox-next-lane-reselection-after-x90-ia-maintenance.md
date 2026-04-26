# X-91: Non-Graybox Next-Lane Reselection After X-90 I-A Maintenance

> Date: 2026-04-18
> Owner: Researcher
> Context: X-90 I-A maintenance完成，所有higher-layer文档已同步到X-89状态

## Question

After X-90 I-A maintenance, what is the next honest non-graybox lane?

## Current Control-Plane State

- `active_gpu_question = none`
- `next_gpu_candidate = none`
- G1-A: deferred-needs-assets (TMIA-DM 512-sample gap)
- B-M0 Candidate A: hold-review-only (CPU-bound shadow-LR)
- I-A maintenance: completed (all higher-layer docs synced to X-89)

## Candidate Pool Review

### Innovation Tracks Status
- **I-A**: All gates completed, higher-layer wording synced
- **I-B**: actual bounded falsifier, no successor (X-54/X-65/X-66)
- **I-C**: translated-contract-only + negative falsifier, no successor (X-56)
- **I-D**: bounded packet + negative rerun, no successor (X-36)

### ROADMAP §5.6 Near-term (2 weeks) Goal
"Force one explicit successor-or-freeze decision on whichever of I-B/I-C/I-D currently has the most plausible fresh hypothesis surface"

## Reselection Logic

Per ROADMAP §5.7: "Expand instead of resting when a track closes cleanly"

**Current situation**:
- I-A track closed cleanly (all maintenance complete)
- No GPU candidates available (G1-A deferred, B-M0 hold-review-only)
- I-B/I-C/I-D all reviewed for successors (X-54/X-56/X-36), all negative

**Options**:
1. Force another successor review on I-B/I-C/I-D (per ROADMAP §5.6)
2. Expand to new candidate surface
3. Continue I-A as stable sidecar

**Honest reading**: 
- I-B/I-C/I-D已在X-54/X-56/X-36中reviewed，均无successor
- 短期内重复review不会产生新hypothesis
- 当前最诚实的move是承认：在genuinely new bounded hypothesis出现前，继续CPU-first维护

## Verdict

**Next honest lane: X-91 = Continue I-A CPU sidecar, monitor for new bounded hypothesis**

**Rationale**:
1. I-A maintenance完成，track处于stable状态
2. I-B/I-C/I-D已reviewed (X-54/X-56/X-36)，短期无successor
3. G1-A/B-M0均非GPU问题，无新GPU候选
4. Per ROADMAP §5.7 "Hold or kill early when only same-family micro-optimization is left"
5. 当前最诚实的posture：保持I-A CPU sidecar，等待genuinely new bounded hypothesis

**Alternative if new hypothesis appears**:
- I-B: genuinely new localization-defense hypothesis
- I-C: genuinely new cross-box support/disconfirm hypothesis  
- I-D: genuinely new conditional successor hypothesis
- New candidate surface expansion

## Next Step

Update ROADMAP control-plane state: current live lane = I-A CPU sidecar (stable), monitor for new bounded hypothesis.
