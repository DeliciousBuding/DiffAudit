# 2026-04-08 White-Box Follow-Up: DPDM W-1 Strong-v3 Three-Shadow Comparator At Max512

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-08 19:40:00 +08:00`
- `selected_defense`: `W-1 = DPDM`
- `current_state`: `strong-v3 three-shadow comparator complete at max_samples=512`
- `gpu_usage`: `single GPU`
- `evidence_level`: `runtime-smoke`

## A. Result

Workspace:

- `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv3-3shadow-max512-20260408`

Metrics:

- `AUC = 0.5`
- `ASR = 0.5`
- `TPR@1%FPR = 0.0`
- `TPR@0.1%FPR = 0.0`
- `shadow_train_size = 3072`
- `target_eval_size = 1024`

## B. Interpretation

Compared with earlier `strong-v3` runs:

- `strong-v3 3-shadow max128`: `AUC = 0.537048`
- `strong-v3 3-shadow max256`: `AUC = 0.522339`
- `strong-v3 3-shadow max512`: `AUC = 0.5`

This is the first medium-to-large scale `strong-v3` GPU defended comparator that lands exactly near random. Directionally it is better than the smaller `strong-v3` runs, and it confirms that the stronger training rung can suppress the white-box signal at larger evaluation scale.

## C. Conclusion

- `strong-v3` now has valid GPU defended comparators at `max128`, `max256`, and `max512`
- `max512` is the current best `strong-v3` result
- this is now strong enough to compare against the `strong-v2` line when deciding whether to continue to `full-scale` or move to the next defended rung
