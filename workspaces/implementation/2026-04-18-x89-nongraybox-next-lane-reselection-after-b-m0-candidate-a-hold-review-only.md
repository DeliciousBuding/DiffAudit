# X-89: Non-Graybox Next-Lane Reselection After B-M0 Candidate A Hold-Review-Only

> Date: 2026-04-18
> Owner: Researcher
> Context: B-M0 Candidate A completed with hold-review-only decision (CPU-bound evaluation)

## Question

After B-M0 Candidate A hold-review-only decision, what is the next honest non-graybox lane?

## Current Control-Plane State

- `active_gpu_question = none`
- `next_gpu_candidate = none` (both G1-A and B-M0 Candidate A are CPU-bound)
- G1-A status: `deferred-needs-assets` (TMIA-DM 512-sample gap)
- B-M0 Candidate A status: `hold-review-only` (CPU-bound evaluation, not GPU question)
- Root ROADMAP B-M0 24h kill gate: CPU-first scoping complete, both candidates are CPU-bound

## Root ROADMAP B-M0 Context Review

**B-M0 Candidate A**: hold-review-only (CPU-bound shadow-LR evaluation)
**B-M0 Candidate B**: I-C in-model white-gray bridge on frozen matched pair
- Blocker: §5.3 blocker B requires `in-model intervention surface` (not offline tensor masking)
- 24h gate: only if blocker B resolves within 24h

**Current honest reading**:
- Both B-M0 candidates are CPU-bound evaluations, not GPU questions
- Root ROADMAP §4 B-M0 intended to release 1 bounded GPU candidate
- Current outcome: no GPU release from B-M0 (both candidates CPU-bound)
- Root ROADMAP §6.2 kill gate: "If 24h CPU-first scoping does not give non-ambiguous GPU choice, hold"

## Verdict

**Next honest lane: Return to I-A higher-layer boundary maintenance (CPU sidecar)**

**Rationale**:
1. Root ROADMAP B-M0 24h kill gate triggered: no GPU release from CPU-first scoping
2. Both G1-A and B-M0 Candidate A classified as CPU-bound evaluations
3. B-M0 Candidate B has 24h blocker (in-model intervention surface not implemented)
4. No other blocked/hold branches have honest reopen (X-54/X-56/X-36 all negative)
5. I-A higher-layer boundary maintenance is current CPU sidecar per ROADMAP §4.5

**X-89 scope**:
- Continue I-A higher-layer boundary maintenance
- Monitor for new bounded hypothesis in I-B/I-C/I-D
- Keep `active_gpu_question = none` until genuinely new bounded hypothesis appears
- Root ROADMAP B-M0 window closes (no GPU release)

**Alternative paths**:
- If B-M0 Candidate B blocker resolves within 24h: reopen B-M0 Candidate B review
- If new bounded hypothesis appears in I-B/I-C/I-D: reopen candidate-surface expansion
- Otherwise: continue I-A CPU sidecar maintenance

## Next Step

Continue I-A higher-layer boundary maintenance (CPU sidecar) per ROADMAP §4.5.
