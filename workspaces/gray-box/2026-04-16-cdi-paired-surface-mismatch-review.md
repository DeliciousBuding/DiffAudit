# 2026-04-16 CDI Paired-Surface Mismatch Review

## Question

Is the new `SecMI 2048` weakness on the paired `CDI` surface a genuine scale effect, a subset-contract mismatch, or an export/config drift?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/secmi-pia-disagreement-20260415-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/secmi-pia-disagreement-20260415-r1/outputs/disagreement_analysis.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/secmi-pia-disagreement-20260415-r1/analysis.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/secmi-cifar10-gpu-full-stat-20260415-r2/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/secmi-pia-disagreement-20260416-r2/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260416-gpu-2048-cdi-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/scripts/run_secmi_pia_disagreement.py`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/secmi-pia-disagreement-20260416-r2/secmi_member_scores.npy`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/secmi-pia-disagreement-20260416-r2/secmi_nonmember_scores.npy`

## Review

### 1. This is not explained by the extra second half alone

- the old paired `1024` surface stayed strong:
  - `SecMI stat AUC = 0.884247`
  - `combined Spearman = 0.907581`
- the new `2048` surface is weak on both halves:
  - first `1024` prefix inside the new run: `AUC = 0.571909`
  - second `1024` suffix inside the new run: `AUC = 0.566015`

Interpretation:

- if the issue were mainly “the added second half is harder,” the first `1024` prefix should have remained near the old `0.884` reference
- it did not
- so `scale truth` is not the primary explanation

### 2. The strongest live suspect is export/config drift

The old strong references used the `SecMI` mainline contract:

- full-split local mainline: `t_sec = 100`
- old strong paired `1024` analyzer: `T_SEC = 100`

The new weak `2048` export changed that contract:

- `run_secmi_pia_disagreement.py` defaults to `t_sec = 20`
- `secmi-pia-disagreement-20260416-r2/summary.json` records `t_sec = 20`

This is not a cosmetic change:

- it changes the observed diffusion depth
- it changes the score source relative to the admitted `SecMI` mainline
- and it is large enough to explain why the same run's first `1024` prefix no longer matches the established `1024` truth

### 3. Subset mismatch is currently secondary, not primary

- the earlier `1024` packet explicitly recorded identical CIFAR-10 split indices between `PIA` and `SecMI`
- the new run still slices deterministic prefixes from the same local member/non-member split roots
- no evidence in this review shows a newly introduced index misalignment

Current honest reading:

- subset-contract mismatch is not fully ruled out
- but it is not needed to explain the failure
- `export/config drift` already explains the dominant mismatch

## Verdict

- `cdi_paired_surface_mismatch_verdict = negative but clarifying`
- primary cause: `export/config drift`
- strongest concrete drift: `t_sec = 20` in the new `2048` export versus the admitted `SecMI` contract at `t_sec = 100`
- `scale truth` is not proven by the current evidence
- `subset mismatch` remains a secondary check item, not the main diagnosis

## Carry-Forward Rule

- do not promote paired `CDI` from the current `2048` `SecMI` export
- keep the existing `1024` paired packet as the last aligned paired reference
- keep the weak `2048` packet as mismatch evidence, not promotion evidence
- `gpu_release = none`

## Next Step

- `SecMI paired-surface repair contract review`

That review should freeze:

1. whether paired `SecMI` export must realign to `t_sec = 100`
2. whether the paired export should follow the official `SecMI` mainline scoring contract exactly
3. whether a future `2048` rerun is justified after contract repair

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
