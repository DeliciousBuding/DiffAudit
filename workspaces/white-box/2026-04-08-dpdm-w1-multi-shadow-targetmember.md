# 2026-04-08 White-Box Follow-Up: DPDM W-1 Multi-Shadow Comparator On Defended Target

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-08 06:58:00 +08:00`
- `selected_defense`: `W-1 = DPDM`
- `current_state`: `multi-shadow comparator complete on defended target-member checkpoint`
- `gpu_usage`: `single GPU`
- `evidence_level`: `runtime-smoke`

## A. Result

Workspace:

- `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-20260408`

Metrics:

- `AUC = 0.496338`
- `ASR = 0.5`
- `TPR@1%FPR = 0.015625`
- `TPR@0.1%FPR = 0.0`
- `shadow_member_mean_score = 0.626829`
- `shadow_nonmember_mean_score = 0.613265`
- `target_member_mean_score = 0.606318`
- `target_nonmember_mean_score = 0.594961`

## B. Interpretation

Compared with the attack mainline:

- `GSA 1k-3shadow`: `AUC = 0.97514`
- `DPDM W-1 multi-shadow comparator on defended target-member checkpoint`: `AUC = 0.496338`

This is the most aligned local `W-1` result currently available in the workspace.

## C. Caveat

This is still not the final benchmark because:

- the defended target and defended shadows are all smoke-scale
- the `DPDM` runs still use `loss.n_noise_samples=1`
- the comparator remains defense-native rather than a direct `GSA` path reuse
