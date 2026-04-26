# 2026-04-16 Post-Temporal-Striding Gray-Box Next-Question Review

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `GB-18 / gray-box lane-status review`
- `question_type`: `post-verdict review`
- `device`: `cpu`
- `decision`: `gray-box currently has no honest new GPU-worthy question`

## Question

After `TMIA-DM late-window + temporal-striding(stride=2)` became the strongest defended gray-box challenger reference, and after `Noise as a Probe` finished its challenger-boundary and defended-extension reviews, does gray-box still contain any honest new GPU-worthy question?

## Executed Evidence

Primary references reviewed:

- defended gray-box challenger update:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-tmiadm-temporal-striding-defense-verdict.md`
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-pia-vs-tmiadm-temporal-striding-defended-comparison.md`
- strengthened bounded challenger candidate boundary:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-noise-as-probe-challenger-boundary-review.md`
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-noise-as-probe-defended-extension-feasibility-review.md`
- gray-box plan / summary state:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\plan.md`
  - `D:\Code\DiffAudit\Research\docs\comprehensive-progress.md`

## Review

### Stable gray-box truths

- `PIA + stochastic-dropout(all_steps)` remains the admitted defended gray-box headline.
- `TMIA-DM late-window + temporal-striding(stride=2)` is now the strongest defended gray-box challenger reference.
- `Noise as a Probe` is stronger than before, but it still remains a `strengthened bounded challenger candidate`, not the packaged active challenger.

### Why the lane does not justify a new GPU question

1. `TMIA-DM temporal-striding` is already decision-grade for the current question.
   - It survived `cpu-32`, `GPU128`, and `GPU256`.
   - The remaining work after that branch was packaging and higher-layer truth sync, which are already done.
   - More same-family rungs would now be mechanical repetition rather than a new bounded hypothesis.

2. `Noise as a Probe` has no honest defended-extension gate on the current contract.
   - Direct `stochastic-dropout` transfer is not a real executable defense on the current `SD1.5` local surface.
   - Direct `temporal-striding` would redefine the attack rather than add a bounded defense.
   - So the branch does not currently expose a defensible next GPU defense task.

3. Existing gray-box side branches are already below release.
   - `PIA + SecMI` naive fusion already closed as `no-go`.
   - `SimA` remained too weak after feasibility and rescan.
   - `structural memorization` remained direction-negative on the current faithful approximation.

## Verdict

Current verdict:

- `GB-18` closes as `negative but stabilizing`

Meaning:

- gray-box is currently strong enough at the story level:
  - `PIA = admitted headline`
  - `TMIA + temporal-striding = strongest defended challenger reference`
  - `Noise as a Probe = strengthened bounded challenger candidate`
- but gray-box currently has `no-new-gpu-question`
- the next honest gray-box GPU release would require:
  - a genuinely new mechanism, or
  - a materially different contract change,
  - not more rungs on already-closed branches

## Carry-Forward Rule

- keep `gpu_release = none`
- keep `next_gpu_candidate = none`
- do not reopen:
  - more `TMIA temporal-striding` scale repeats
  - `TMIA` fusion retries
  - direct `Noise as a Probe` defended-extension retries on the current local contract
- if gray-box is revisited, prefer:
  - a genuinely new family, or
  - a real contract/provenance change that creates a new bounded question

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `docs/comprehensive-progress.md`: no additional wording change required for this step
- `Platform`: no handoff required
- `Runtime`: no handoff required
- `competition materials`: no immediate sync required
