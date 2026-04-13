# 2026-04-08 White-Box Follow-Up: DPDM W-1 Strong-v2 Multi-Shadow Comparator

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-08 07:40:00 +08:00`
- `selected_defense`: `W-1 = DPDM`
- `current_state`: `strong-v2 multi-shadow comparator complete on defended target + defended shadows`
- `gpu_usage`: `single GPU`
- `evidence_level`: `runtime-smoke`

## A. Result

Workspace:

- `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv2-20260408`

Metrics:

- `AUC = 0.541199`
- `ASR = 0.550781`
- `TPR@1%FPR = 0.0`
- `TPR@0.1%FPR = 0.0`
- `shadow_member_mean_score = 0.791869`
- `shadow_nonmember_mean_score = 0.742501`
- `target_member_mean_score = 0.767139`
- `target_nonmember_mean_score = 0.727575`

## B. Interpretation

Compared with the attack mainline:

- `GSA 1k-3shadow`: `AUC = 0.97514`
- `DPDM W-1 defended-target + defended-shadows strong-v2 comparator`: `AUC = 0.541199`

This is the strongest local `W-1` comparator currently available in the workspace.

## C. Execution Notes

- The earlier shadow watcher bug was confirmed to be a run-suffix propagation issue, not a training failure.
- `launch_dpdm_shadow_sequence.ps1` now accepts `RunSuffix`, so defended shadows no longer fall back into the `smoke-v1` namespace.
- The defended target and both defended shadows now exist as independent `strong-v2` checkpoints.

## D. Caveat

This is still not the final benchmark because:

- the comparator remains defense-native rather than a direct `GSA` path reuse
- the current result is still tagged `runtime-smoke`, not `benchmark-ready`
- the next bottleneck is evaluation scale and benchmark alignment, not checkpoint existence
