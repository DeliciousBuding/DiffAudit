# 2026-04-16 DP-LoRA Local Evaluator Hardening

## Question

After `WB-15` showed that secondary-metric harmonization is blocked by frozen-output insufficiency and an unseeded split, can the local evaluator now be hardened so future rerenders become audit-safe?

## Inputs

- `workspaces/white-box/2026-04-16-dplora-secondary-metric-harmonization-audit.md`
- `scripts/evaluate_smp_lora_defense.py`

## Changes

The local evaluator is now hardened in three ways:

1. deterministic split control
   - `evaluation_seed` is now explicit
   - the train/test permutation now uses a fixed RNG seeded from that value
2. richer defended-style metrics
   - `ASR`
   - `TPR@1%FPR`
   - `TPR@0.1%FPR`
   are now emitted alongside `accuracy` and `auc`
3. score-artifact persistence
   - CLI callers can now write a companion details artifact containing:
     - `y_test`
     - `y_pred`
     - `y_prob`
     - `fpr`
     - `tpr`

## Verification

- `conda run -n diffaudit-research python -m py_compile scripts/evaluate_smp_lora_defense.py`
- result: `pass`

## Interpretation

This does **not** yet harmonize the local board by itself.

It does:

- remove the evaluator-side blocker identified in `WB-15`
- make the next local rerender capable of producing defended-style secondary metrics
- make future local outputs more auditable and replayable

## Verdict

- `local_evaluator_hardening = positive`
- `deterministic_split_control = added`
- `secondary_metrics_support = added`
- `score_artifact_persistence = added`
- `harmonized_local_board_exists = no`
- `gpu_release = none`
- `next_step = decide whether to rerender baseline and frozen SMP-LoRA under the hardened evaluator`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this is execution-layer hardening inside the research lane only.
