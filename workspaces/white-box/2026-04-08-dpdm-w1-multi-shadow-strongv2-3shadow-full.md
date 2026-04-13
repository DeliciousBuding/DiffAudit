# 2026-04-08 White-Box Follow-Up: DPDM W-1 Strong-v2 Three-Shadow Full-Scale Comparator

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-08 09:24:00 +08:00`
- `selected_defense`: `W-1 = DPDM`
- `current_state`: `strong-v2 three-shadow full-scale comparator complete`
- `gpu_usage`: `single GPU`
- `evidence_level`: `runtime-smoke`

## A. Result

Workspace:

- `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv2-3shadow-full-20260408`

Metrics:

- `AUC = 0.490813`
- `ASR = 0.496`
- `TPR@1%FPR = 0.006`
- `TPR@0.1%FPR = 0.0`
- `shadow_train_size = 6000`
- `target_eval_size = 2000`

## B. Interpretation

Compared with prior defended comparators:

- `strong-v2` at `2-shadow max512`: `AUC = 0.537201`
- `strong-v2` at `3-shadow max512`: `AUC = 0.462799`
- `strong-v2` at `3-shadow full-scale`: `AUC = 0.490813`

The full-scale run rises slightly above the `3-shadow max512` result, but it remains far below the `GSA 1k-3shadow` attack mainline. This is still a defensible local white-box defense result rather than a collapse back toward the attack baseline.

## C. Conclusion

- `GSA 1k-3shadow` remains the attack upper bound with `AUC = 0.97514`
- `W-1 strong-v2 3-shadow full-scale` stays near random with `AUC = 0.490813`
- the next useful increment is no longer more evaluation scale on the same checkpoint, but a stronger defended training rung
