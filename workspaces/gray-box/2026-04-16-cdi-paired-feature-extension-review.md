# 2026-04-16 CDI Paired-Feature Extension Review

## Question

Given the completed `PIA 2048` surface and the new `SecMI 2048` paired-surface verdict, should the `CDI` lane promote immediately into paired-feature scoring, or should it pause and review the mismatch first?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-cdi-feature-collection-surface-review.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-cdi-internal-canary-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-pia-2048-cdi-rung-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-secmi-pia-2048-paired-surface-verdict.md`

## Review

### What is now strong

- `CDI` already has:
  - a frozen first contract
  - a real first internal canary
  - a larger reusable `PIA` score surface

### What is now weak

- the widened `SecMI 2048` paired surface is not stable enough yet for immediate feature promotion:
  - `SecMI stat AUC` dropped to near-boundary quality
  - cross-method agreement weakened substantially
  - disagreement increased, but not in a clearly exploitable way

### Therefore

- immediate paired-feature promotion would currently overclaim
- the next honest step is review / diagnosis, not scorer escalation

## Verdict

- `cdi_paired_feature_extension_review = negative but useful`
- do **not** promote the lane yet into paired `PIA + SecMI` feature scoring
- keep:
  - `CDI first canary = landed`
  - `paired extension = review-required`
- `gpu_release = none` until the `SecMI 2048` mismatch is explained or bounded away

## Next Step

- `CDI paired-surface mismatch review`

That review should answer:

1. whether the `SecMI 2048` weakness is:
   - genuine scale behavior
   - subset-contract mismatch
   - export/config drift
2. whether paired `CDI` should fall back to:
   - `1024` aligned surface
   - or a revised `2048` contract

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

