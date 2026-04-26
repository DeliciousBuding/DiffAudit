# 2026-04-16 SecMI-PIA 2048 Repaired Paired-Surface Verdict

## Question

After freezing the repaired `SecMI` export contract, does the `2048` paired surface recover enough strength to replace the earlier weak drifted packet as the active `CDI` paired reference?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-secmi-paired-surface-repair-contract-review.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\secmi-pia-disagreement-20260416-r2\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\secmi-pia-disagreement-20260416-r3\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\secmi-pia-disagreement-20260415-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-cifar10-runtime-mainline-20260416-gpu-2048-cdi-r1\summary.json`

## Results

Repaired run root:

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\secmi-pia-disagreement-20260416-r3`

Repaired metrics:

- `secmi_stat_auc = 0.876912`
- `secmi_stat_asr = 0.808594`
- `combined_spearman = 0.906879`
- `topk_overlap = 0.617647`
- `disagreement_rate = 0.121582`
- `t_sec = 100`

Comparison:

- weak drifted `2048` packet (`r2`):
  - `secmi_stat_auc = 0.569096`
  - `combined_spearman = 0.677189`
  - `disagreement_rate = 0.266602`
  - `t_sec = 20`
- old strong `1024` packet:
  - `secmi_stat_auc = 0.884180`
  - `combined_spearman = 0.907588`
  - `disagreement_rate = 0.122559`

## Interpretation

1. The repaired `2048` paired surface is no longer weak.
2. Once the export returns to the admitted `SecMI` contract, the paired metrics return to the same quality regime as the earlier strong `1024` packet.
3. The old weak `2048` result should now be treated as drift evidence, not as the active truth of the paired surface.

## Verdict

- `secmi_pia_2048_repaired_paired_surface_verdict = positive`
- the repaired `2048` paired surface is now strong enough to become the active `CDI` paired reference
- the old `r2` packet is demoted to mismatch-history evidence
- no further immediate GPU rerun is justified from this question alone

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
