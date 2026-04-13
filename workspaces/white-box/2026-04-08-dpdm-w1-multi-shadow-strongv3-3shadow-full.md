# 2026-04-08 White-Box Follow-Up: DPDM W-1 Strong-v3 Three-Shadow Full-Scale Comparator

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-08 21:05:00 +08:00`
- `selected_defense`: `W-1 = DPDM`
- `current_state`: `strong-v3 three-shadow full-scale comparator complete`
- `gpu_usage`: `single GPU`
- `evidence_level`: `runtime-smoke`

## A. Result

Workspace:

- `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv3-3shadow-full-rerun8-20260408`

Metrics:

- `AUC = 0.488783`
- `ASR = 0.4985`
- `TPR@1%FPR = 0.009`
- `TPR@0.1%FPR = 0.0`
- `shadow_train_size = 6000`
- `target_eval_size = 2000`

## B. Interpretation

Compared with earlier `strong-v3` runs:

- `strong-v3 3-shadow max128`: `AUC = 0.537048`
- `strong-v3 3-shadow max256`: `AUC = 0.522339`
- `strong-v3 3-shadow max512`: `AUC = 0.5`
- `strong-v3 3-shadow full-scale`: `AUC = 0.488783`

This is the strongest completed `strong-v3` defended comparator so far. It is now slightly better than the `strong-v2 3-shadow full-scale` comparator and is the current best `strong-v3` evidence line.

## C. Conclusion

- `strong-v3` now has valid GPU defended comparators at `max128`, `max256`, `max512`, and `full-scale`
- `full-scale` is the current best `strong-v3` result
- this is now strong enough to stand beside the `strong-v2` line in the white-box comparison table
