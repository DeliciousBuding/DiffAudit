# 2026-04-08 White-Box Follow-Up: DPDM W-1 Strong-v3 Three-Shadow Comparator At Max256

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-08 18:30:00 +08:00`
- `selected_defense`: `W-1 = DPDM`
- `current_state`: `strong-v3 three-shadow comparator complete at max_samples=256`
- `gpu_usage`: `single GPU`
- `evidence_level`: `runtime-smoke`

## A. Result

Workspace:

- `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv3-3shadow-max256-20260408`

Metrics:

- `AUC = 0.522339`
- `ASR = 0.527344`
- `TPR@1%FPR = 0.003906`
- `TPR@0.1%FPR = 0.0`
- `shadow_train_size = 1536`
- `target_eval_size = 512`

## B. Interpretation

Compared with the earlier `strong-v3 max128` result:

- `strong-v3 3-shadow max128`: `AUC = 0.537048`
- `strong-v3 3-shadow max256`: `AUC = 0.522339`

This is the first medium-scale stable GPU defended comparator on the `strong-v3` rung. It remains far below the `GSA` attack mainline and is more useful than the CPU diagnostic for judging whether `strong-v3` is worth pushing further.

## C. Decision Signal

- `strong-v3` is not just “GPU executable once”; it now has stable results at both `max128` and `max256`
- the next decision is whether to spend GPU on `max512`, not whether the rung is valid
