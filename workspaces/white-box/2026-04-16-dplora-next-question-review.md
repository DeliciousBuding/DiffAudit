# 2026-04-16 DP-LoRA Next-Question Review

## Question

After `WB-13` refreshed the comparator admission packet around the completed local board, does this lane still contain a new bounded question, or should it now explicitly enter `no-new-gpu-question` hold?

## Inputs

- `workspaces/white-box/2026-04-16-dplora-local-board-refresh-verdict.md`
- `workspaces/white-box/2026-04-16-dplora-comparator-release-review-refresh.md`
- `workspaces/intake/2026-04-16-dplora-comparator-admission-packet-refresh.md`
- `scripts/evaluate_smp_lora_defense.py`
- `outputs/smp-lora-phase2/baseline_nodefense_target-64/eval_baseline.py`

## What Is Already Answered

The following questions are now closed:

1. does a same-asset local comparator board exist?
   - `yes`
2. does the local board favor frozen `SMP-LoRA` over refreshed `W-1` on the shared primary metric?
   - `yes`
3. does that local-board win automatically release a new GPU task?
   - `no`

So repeating:

- optimizer/lr rescue
- another local-board rerun with no new hypothesis
- another `W-1` local63 rerender with no new question

would all be stale.

## Remaining Gap

One bounded gap still remains:

- the local `baseline / SMP-LoRA` rows expose only:
  - `accuracy`
  - `auc`
- while the refreshed `W-1` row also exposes:
  - `ASR`
  - `TPR@1%FPR`
  - `TPR@0.1%FPR`

So the board is now aligned on the primary metric, but still not aligned on defended-style secondary metrics.

## Feasibility Read

This looks like a real next question, and it is CPU-first.

Why:

1. `SimpleGSA.evaluate(...)` already computes:
   - `y_pred`
   - `y_prob`
2. it already imports:
   - `accuracy_score`
   - `roc_auc_score`
3. so extending it to also emit:
   - `ASR` as explicit attack success rate / held-out accuracy
   - `TPR@1%FPR`
   - `TPR@0.1%FPR`
   is a small evaluation-layer change, not a new research family
4. that change would still require rerendering the local `baseline / SMP-LoRA` outputs, but it does not require inventing a new GPU hypothesis first

## Decision

So the lane does still contain one bounded next question:

- `secondary-metric harmonization on the existing local board`

But it does **not** contain:

- an honest new GPU release question
- an honest new training sweep
- an honest admitted-upgrade question

## Verdict

- `next_question_review = positive but narrow`
- `new_bounded_question_exists = yes`
- `next_question = local-board secondary-metric harmonization`
- `next_question_type = cpu-first evaluation-layer hardening`
- `new_gpu_question_exists = no`
- `gpu_release = none`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this updates white-box queue discipline only.
