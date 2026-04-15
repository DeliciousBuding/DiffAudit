# 2026-04-16 Gray-Box Second-Defense Shortlist Review

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `GB-1 review / expand`
- `decision_grade`: `roadmap-directing`
- `gpu_status`: `none`

## Question

After multiple bounded gray-box defense candidate smokes, is there still a low-cost second-defense question worth keeping on the immediate execution path, or should gray-box effort pivot elsewhere?

## Reviewed Shortlist

Current defended mainline:

- `stochastic-dropout(all_steps)`
  - status: `positive provisional G-1`

Rejected bounded candidates:

1. `epsilon-precision-throttling`
   - verdict: `negative but useful`
2. `epsilon-output-noise`
   - verdict: `negative but useful`
3. `input-gaussian-blur`
   - verdict: `negative`

Still-blocked larger candidate:

4. `G-2 distillation proxy model`
   - repo state:
     - `blocked`
   - reason:
     - no formal distillation training chain
     - no unified evaluation packet
     - not a bounded next-step implementation under the current repo state

## Review Conclusion

Current shortlist verdict:

- `cheap perturbation frontier exhausted`

Interpretation:

1. the repo now has multiple bounded negative results for small output-side and input-side perturbation defenses;
2. continuing to try nearby perturbation knobs is more likely to repeat the same mistake than to open a genuinely new defended line;
3. the only materially different gray-box defense candidate named in the defense index, `G-2 distillation`, is still execution-blocked rather than merely untested.

## Decision

Immediate execution policy:

1. do not open another low-cost perturbation-style `GB-1` smoke without a fresh hypothesis;
2. do not release any gray-box GPU task for these rejected perturbation candidates;
3. keep `GB-1` open at the roadmap level, but treat its next executable step as:
   - either `G-2` unblock/design work
   - or a pivot to `GB-3` new-family exploration if mainline momentum matters more

## Recommended Next Task

Recommended next active gray-box task:

- `GB-3.1` choose one new family

Recommended first family:

- `SimA`

Reason:

1. it is already in the gray-box signal-axis framing;
2. it is more materially different from `PIA` than another defense-side perturbation knob;
3. it offers a cleaner path to method diversity than forcing a blocked `G-2` training program immediately.

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: no direct change needed.
- Materials: gray-box wording can now honestly say the repo tried multiple cheap second-defense candidates and rejected them, which strengthens the case for shifting gray-box effort toward new-family diversity instead of more near-duplicate defense knobs.
