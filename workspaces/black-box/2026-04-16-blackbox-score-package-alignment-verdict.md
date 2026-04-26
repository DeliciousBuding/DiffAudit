# 2026-04-16 Black-Box Score-Package Alignment Verdict

## Task

- `BB-6.1` align `Recon` and `semantic-aux` score artifacts under one same-protocol comparison contract

## Question

- Can the current admitted `Recon` artifacts and current `semantic-auxiliary-classifier` challenger artifacts already be treated as one same-protocol score-package surface, or is there still a protocol/asset mismatch that blocks an honest direct package?

## Evidence Base

- `workspaces/black-box/2026-04-10-recon-decision-package.md`
- `workspaces/black-box/2026-04-15-blackbox-second-signal-semantic-aux-verdict.md`
- `workspaces/black-box/plan.md`
- `workspaces/black-box/2026-04-15-clid-local-crosscheck-note.md`

## Result

Current artifacts do **not** yet share one clean same-protocol surface.

Why:

1. admitted `Recon` headline artifacts are frozen around:
   - `public-100 step30`
   - `proxy-shadow-member`
   - `derived-public-*`
   - `celeba_partial_target`
2. current `semantic-auxiliary-classifier` challenger artifacts are frozen around:
   - local CelebA target-family comparator
   - current target-family image stack
   - `celeba_target/checkpoint-25000`
3. this means the two lines differ at both:
   - split semantics
   - target checkpoint / asset contract

## Verdict

- `negative but useful`

The new `BB-6` lane is real, but `BB-6.1` cannot honestly claim alignment on existing artifacts alone. The blocker is not missing creativity; it is contract mismatch.

## Decision

Current decision:

- close `BB-6.1` as `negative but useful`
- do not test `BB-6.2 / BB-6.3` on the current mixed artifact set
- keep `gpu_release = none`
- if `BB-6` continues, the next bounded step must first create one aligned semantic-aux comparator on the same `Recon` contract surface

## Handoff Note

- `Platform`: no sync needed.
- `Runtime`: no sync needed.
- `Materials`: do not describe `Recon + semantic-aux` as a current package yet; the protocol surface is not aligned.
