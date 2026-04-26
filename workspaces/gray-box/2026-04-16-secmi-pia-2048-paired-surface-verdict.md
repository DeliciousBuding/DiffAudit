# 2026-04-16 SecMI-PIA 2048 Paired-Surface Verdict

## Question

After the new `PIA 2048` shared-score surface landed, can `SecMI stat` be widened onto the same `2048` subset cleanly enough to support the next `CDI paired-feature extension` step?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-cifar10-runtime-mainline-20260416-gpu-2048-cdi-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-cifar10-runtime-mainline-20260416-gpu-2048-cdi-r1\scores.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\secmi-pia-disagreement-20260415-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\secmi-pia-disagreement-20260416-r2\summary.json`

## What Landed

Run root:

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\secmi-pia-disagreement-20260416-r2`

Observed metrics:

- `secmi_stat_auc = 0.569096`
- `secmi_stat_asr = 0.552002`
- `member_spearman = 0.836301`
- `nonmember_spearman = 0.6865`
- `combined_spearman = 0.677189`
- `topk_overlap = 0.426471`
- `disagreement_rate = 0.266602`
- `secmi_orientation = negated`

## Comparison Against 1024 Reference

Reference paired surface:

- `secmi-pia-disagreement-20260415-r1`
- `secmi_stat_auc = 0.884180`
- `combined_spearman = 0.907588`
- `disagreement_rate = 0.122559`

Interpretation:

1. The `2048` paired surface does not preserve the strong `SecMI stat` behavior seen on the earlier `1024` paired subset.
2. Correlation remains real, but it is materially weaker.
3. Disagreement roughly doubles, but not in a healthy “two strong complementary rankers” way.
4. Current honest reading is not “paired surface improved”; it is “paired surface destabilized”.

## Verdict

- `secmi_pia_2048_paired_surface_verdict = mixed but useful`
- the widened paired surface is execution-positive
- but it is **not** currently good enough to justify immediate paired `CDI` scorer promotion
- the mainline consequence is:
  - keep `CDI first canary = SecMI stat only`
  - treat `PIA + SecMI` on `2048` as a review target, not a promotion-ready extension

## Carry-Forward Rule

- do not open another GPU scaling rung immediately from this result
- the next honest move is CPU-side review:
  - decide whether the `SecMI 2048` degradation is a scale truth worth preserving, or a contract bug worth isolating

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

