# 2026-04-08 White-Box Follow-Up: DPDM W-1 Strong-v2 Multi-Shadow Comparator At Max512

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-08 08:12:00 +08:00`
- `selected_defense`: `W-1 = DPDM`
- `current_state`: `strong-v2 two-shadow comparator complete at max_samples=512`
- `gpu_usage`: `single GPU`
- `evidence_level`: `runtime-smoke`

## A. Result

Workspace:

- `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv2-max512-20260408`

Metrics:

- `AUC = 0.537201`
- `ASR = 0.530273`
- `TPR@1%FPR = 0.013672`
- `TPR@0.1%FPR = 0.0`
- `shadow_train_size = 2048`
- `target_eval_size = 1024`

## B. Interpretation

Compared with the smaller defended comparator:

- `strong-v2` at `max_samples=128`: `AUC = 0.541199`
- `strong-v2` at `max_samples=512`: `AUC = 0.537201`

The larger evaluation scale does not collapse the defense result back toward the `GSA` attack mainline. This makes the current `W-1` direction more defensible than a tiny-sample-only claim.

## C. Next Step

- `shadow-01 strong-v2` has been queued and launched after this run.
- the next comparator should align to `1k-3shadow` by using three defended shadows under the same `max_samples=512` setting.
