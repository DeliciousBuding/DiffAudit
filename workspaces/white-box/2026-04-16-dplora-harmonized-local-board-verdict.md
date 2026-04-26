# 2026-04-16 DP-LoRA Harmonized Local Board Verdict

## Question

After `WB-16` hardened the local evaluator, what does the first harmonized local comparator board now say about `baseline vs frozen SMP-LoRA vs W-1`?

## Inputs

- `outputs/smp-lora-phase2/baseline_nodefense_target-64-harmonized-20260416/evaluation.json`
- `outputs/smp-lora-sweep/lambda0.1_rank4_ep10/evaluation_pretrained_harmonized_20260416.json`
- `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv3-3shadow-local63-20260416/summary.json`
- `workspaces/white-box/2026-04-16-dplora-local-board-refresh-verdict.md`
- `workspaces/white-box/2026-04-16-dplora-local-evaluator-hardening.md`

## Harmonized Local Board

### Baseline local63 rerender

- `AUC = 0.4061624649859944`
- `ASR = 0.5263157894736842`
- `TPR@1%FPR = 0.0`
- `TPR@0.1%FPR = 0.0`
- `evaluation_seed = 42`

### Frozen SMP-LoRA local63 rerender

- `AUC = 0.43137254901960786`
- `ASR = 0.42105263157894735`
- `TPR@1%FPR = 0.0`
- `TPR@0.1%FPR = 0.0`
- `evaluation_seed = 42`

### W-1 strong-v3 local63

- `AUC = 0.474175`
- `ASR = 0.484127`
- `TPR@1%FPR = 0.0`
- `TPR@0.1%FPR = 0.0`

## Ordering

On `AUC` alone, lower is better for the defender:

- `baseline = 0.406162...`
- `SMP-LoRA = 0.431373...`
- `W-1 = 0.474175`

On `ASR`, lower is also better for the defender:

- `SMP-LoRA = 0.421053...`
- `W-1 = 0.484127`
- `baseline = 0.526316...`

On low-FPR point metrics:

- all three rows are tied at `0.0`

## Interpretation

This means the old simple local ordering from `WB-11` no longer holds under the hardened evaluator.

What survives:

1. frozen `SMP-LoRA` still beats refreshed `W-1` on both:
   - `AUC`
   - `ASR`
2. the local board is now more honest than the old unseeded read

What changes:

1. `baseline` is now better than frozen `SMP-LoRA` on `AUC`
2. so the local board is no longer a one-line `SMP-LoRA > W-1 > baseline` story
3. it is now a `metric-split local board`

## Boundary

This still does **not** imply:

- admitted white-box upgrade
- full-scale comparator replacement
- automatic new GPU release

## Verdict

- `harmonized_local_board_verdict = mixed but useful`
- `old_wb11_ordering_still_valid = no`
- `smp_lora_beats_w1_locally = yes`
- `smp_lora_beats_baseline_locally = not-on-all-metrics`
- `current_local_story = metric-split-board`
- `gpu_release = none`
- `next_step = refresh queue truth around the harmonized local board before any further question selection`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this changes internal white-box comparator truth, but still sits below admitted/system-facing claim level.
