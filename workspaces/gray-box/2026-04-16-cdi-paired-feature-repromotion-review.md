# 2026-04-16 CDI Paired-Feature Re-Promotion Review

## Question

Now that the repaired `SecMI 2048` paired surface has recovered, should the `CDI` lane re-open paired `PIA + SecMI` feature promotion?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-cdi-paired-feature-extension-review.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-cdi-paired-surface-mismatch-review.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-secmi-paired-surface-repair-contract-review.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-secmi-pia-2048-repaired-paired-surface-verdict.md`

## Review

- the earlier promotion stop was real and correct at the time:
  - weak `SecMI 2048` packet
  - drift-heavy contract
  - no honest basis for paired extension
- that blocker is now removed:
  - repaired `2048` paired packet is strong again
  - cross-method agreement is back near the old strong `1024` level
  - disagreement is back in the same moderate range rather than the earlier inflated mismatch regime

## Verdict

- `cdi_paired_feature_repromotion_review = positive but bounded`
- paired `PIA + SecMI` feature promotion may now re-open
- this does **not** replace the landed `SecMI-only` first canary
- it does justify the next CPU-side lane:
  - `CDI paired-feature scorer design`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
