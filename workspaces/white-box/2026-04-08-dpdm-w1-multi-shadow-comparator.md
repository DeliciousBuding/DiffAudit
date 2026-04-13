# 2026-04-08 White-Box Follow-Up: DPDM W-1 Multi-Shadow Comparator

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-08 06:45:00 +08:00`
- `selected_defense`: `W-1 = DPDM`
- `current_state`: `multi-shadow comparator complete`
- `gpu_usage`: `single GPU`
- `evidence_level`: `runtime-smoke`

## A. Result

Workspace:

- `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-20260408`

Metrics:

- `AUC = 0.493652`
- `ASR = 0.5`
- `TPR@1%FPR = 0.015625`
- `TPR@0.1%FPR = 0.0`
- `shadow_member_mean_score = 0.626829`
- `shadow_nonmember_mean_score = 0.613265`
- `target_member_mean_score = 4.803496`
- `target_nonmember_mean_score = 4.502493`
- `shadow_train_size = 384`
- `target_eval_size = 128`

## B. Interpretation

This is the first defended multi-shadow white-box comparator in the current workspace.

Relative to the strong `GSA 1k-3shadow` attack result:

- `GSA` achieved `AUC = 0.97514`
- `DPDM W-1 multi-shadow comparator` stays near random at `AUC = 0.493652`

Directionally, this continues to support `W-1` as a leakage-reduction route.

## C. Caveat

This is still not the final white-box defense benchmark because:

- the defended shadows are smoke-scale
- the `DPDM` shadows still use `loss.n_noise_samples=1`
- the comparator is defense-native rather than a direct `GSA` model-path reuse
