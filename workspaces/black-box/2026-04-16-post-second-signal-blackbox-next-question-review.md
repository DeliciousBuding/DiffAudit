# 2026-04-16 Post-Second-Signal Black-Box Next-Question Review

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `BB-7 / black-box lane-status review`
- `question_type`: `post-verdict review`
- `device`: `cpu`
- `decision`: `black-box currently has no honest new GPU-worthy question`

## Question

After `semantic-auxiliary-classifier` landed as the leading black-box challenger, `CLiD` was tightened to `evaluator-near local clip-only corroboration`, `served-image-sanitization` closed as a mitigation no-go, `variation` became `contract-ready blocked`, and the same-protocol `Recon + semantic-aux` score package landed as `positive but bounded`, does black-box still contain any honest new GPU-worthy question?

## Executed Evidence

Primary references reviewed:

- second-signal black-box challenger:
  - `D:\Code\DiffAudit\Research\workspaces\black-box\2026-04-15-blackbox-second-signal-semantic-aux-verdict.md`
  - `D:\Code\DiffAudit\Research\workspaces\black-box\runs\semantic-aux-classifier-comparator-20260416-r2\summary.json`
- challenger scoring and same-protocol package:
  - `D:\Code\DiffAudit\Research\workspaces\black-box\2026-04-16-blackbox-semantic-aux-scoring-verdict.md`
  - `D:\Code\DiffAudit\Research\workspaces\black-box\2026-04-16-blackbox-score-package-verdict.md`
- boundary and blocked branches:
  - `D:\Code\DiffAudit\Research\workspaces\black-box\2026-04-16-clid-threshold-evaluator-compatibility-verdict.md`
  - `D:\Code\DiffAudit\Research\workspaces\black-box\2026-04-16-blackbox-served-image-mitigation-verdict.md`
  - `D:\Code\DiffAudit\Research\workspaces\black-box\2026-04-16-variation-asset-contract-verdict.md`
- black-box plan / roadmap state:
  - `D:\Code\DiffAudit\Research\workspaces\black-box\plan.md`
  - `D:\Code\DiffAudit\Research\ROADMAP.md`

## Review

### Stable black-box truths

- `Recon` remains the frozen black-box headline and best project-level existence-proof package.
- `semantic-auxiliary-classifier` is the leading second-signal black-box challenger.
- `CLiD` remains useful only as `evaluator-near local clip-only corroboration`, not a paper-aligned local benchmark.
- `variation` is blocked by missing real assets, not by missing runner logic.

### Why the lane does not justify a new GPU question

1. The active challenger already has a finished bounded verdict.
   - `semantic-auxiliary-classifier` stayed stable at `32 / 32`.
   - current scoring / calibration upgrades did not create a materially new ranking family.
   - more scale without a new feature family would now be mechanical.

2. The black-box same-protocol package is useful but still local.
   - the aligned `Recon + semantic-aux` package improved actionability on one bounded surface,
   - but it did not replace the frozen `Recon` package and did not open a clean new GPU hypothesis.

3. The remaining branches are boundary- or asset-blocked, not GPU-blocked.
   - `CLiD` needs shadow-side evaluator assets.
   - `variation` needs `query_image_root / query images` and the rest of its explicit unblock contract.
   - `served-image-sanitization` already closed as a meaningful mitigation no-go.

## Verdict

Current verdict:

- `BB-7` closes as `negative but stabilizing`

Meaning:

- black-box is stable at the story level:
  - `Recon = headline`
  - `semantic-auxiliary-classifier = leading challenger`
  - `CLiD = corroboration / boundary-only`
  - `variation = contract-ready blocked`
- but black-box currently has `no-new-gpu-question`
- the next honest black-box reopen should require:
  - a genuinely new feature family, or
  - real blocked assets becoming available,
  - not more mechanical scale-up on current challenger logic

## Carry-Forward Rule

- keep `gpu_release = none`
- keep `next_gpu_candidate = none`
- do not reopen:
  - more semantic-aux scale-up without a new feature family
  - more scoring-only or fusion-only retries on the same ordering
  - `variation` result runs before the asset contract is filled
- if black-box is revisited, prefer:
  - a genuinely new family, or
  - a real asset/boundary change that creates a new bounded question

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/black-box/plan.md`: update required
- `docs/comprehensive-progress.md`: no additional wording change required for this step
- `Platform`: no handoff required
- `Runtime`: no handoff required
- `competition materials`: no immediate sync required
