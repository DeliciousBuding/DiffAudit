# 2026-04-08 White-Box Follow-Up: DPDM W-1 Strong-v2 Three-Shadow Comparator At Max512

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-08 09:02:00 +08:00`
- `selected_defense`: `W-1 = DPDM`
- `current_state`: `strong-v2 three-shadow comparator complete at max_samples=512`
- `gpu_usage`: `single GPU`
- `evidence_level`: `runtime-smoke`

## A. Result

Workspace:

- `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv2-3shadow-max512-20260408`

Metrics:

- `AUC = 0.462799`
- `ASR = 0.47168`
- `TPR@1%FPR = 0.0`
- `TPR@0.1%FPR = 0.0`
- `shadow_train_size = 3072`
- `target_eval_size = 1024`

## B. Interpretation

Compared with earlier defended comparators:

- `strong-v2` at `2-shadow max128`: `AUC = 0.541199`
- `strong-v2` at `2-shadow max512`: `AUC = 0.537201`
- `strong-v2` at `3-shadow max512`: `AUC = 0.462799`

This is the current strongest local `W-1` white-box defense result in the workspace. It is materially below the `GSA 1k-3shadow` attack mainline and is now the closest local defended comparator to the same `1k-3shadow` structure.

## C. Conclusion

- `GSA 1k-3shadow` remains the attack upper bound with `AUC = 0.97514`
- `W-1 strong-v2 3-shadow max512` reduces the defended comparator to `AUC = 0.462799`
- the next bottleneck is no longer “can we run the defended comparator”, but whether to continue increasing benchmark scale or training strength
