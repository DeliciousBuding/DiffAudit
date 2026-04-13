# 2026-04-08 White-Box Follow-Up: DPDM W-1 Strong-v3 Three-Shadow Comparator At Max128

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-08 14:55:00 +08:00`
- `selected_defense`: `W-1 = DPDM`
- `current_state`: `strong-v3 three-shadow comparator complete at max_samples=128`
- `gpu_usage`: `single GPU`
- `evidence_level`: `runtime-smoke`

## A. Result

Workspace:

- `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv3-3shadow-max128-20260408`

Metrics:

- `AUC = 0.537048`
- `ASR = 0.519531`
- `TPR@1%FPR = 0.0`
- `TPR@0.1%FPR = 0.0`
- `shadow_train_size = 768`
- `target_eval_size = 256`

## B. Interpretation

This confirms that the `strong-v3` checkpoint set can produce a valid GPU three-shadow defended comparator when the evaluation scale is reduced.

Compared with prior white-box results:

- `strong-v2 3-shadow full-scale`: `AUC = 0.490813`
- `strong-v3 3-shadow max128`: `AUC = 0.537048`

So `strong-v3` is now a valid local defended rung, but it has not yet surpassed the best `strong-v2` defended comparator.

## C. Diagnostic Meaning

- `strong-v3` asset validity is no longer in question
- the remaining instability is specific to larger-scale GPU comparator runs
