# 2026-04-16 DP-LoRA Secondary-Metric Harmonization Audit

## Question

After `WB-14` selected `local-board secondary-metric harmonization` as the only remaining bounded question, can the current frozen local-board artifacts already be harmonized post hoc, or does the evaluation surface need hardening first?

## Inputs

- `workspaces/white-box/2026-04-16-dplora-next-question-review.md`
- `scripts/evaluate_smp_lora_defense.py`
- `outputs/smp-lora-phase2/baseline_nodefense_target-64/eval_baseline.py`
- `outputs/smp-lora-phase2/baseline_nodefense_target-64/evaluation.json`
- `outputs/smp-lora-sweep/lambda0.1_rank4_ep10/evaluation_pretrained.json`

## What The Current Evaluator Can Already Do

`SimpleGSA.evaluate(...)` already computes:

- `y_pred`
- `y_prob`
- `accuracy`
- `auc`

and therefore could be extended to emit:

- `ASR` as explicit attack success rate / held-out accuracy
- `TPR@1%FPR`
- `TPR@0.1%FPR`

So this is not blocked by model architecture or asset availability.

## What The Frozen Artifacts Still Lack

The current frozen local-board artifacts only preserve:

- headline JSON outputs

They do **not** preserve:

- per-example scores
- classifier probabilities
- ROC-curve coordinates
- explicit evaluation seed
- explicit saved permutation / split

Also, the current evaluator still uses:

- `np.random.permutation(len(X))`

without a fixed recorded seed.

That means a naïve rerender for extra metrics would not be a pure metadata upgrade.

It would implicitly refresh the local board itself.

## Audit Decision

So the current harmonization question is:

- `real`
- but `not yet artifact-safe`

The blocker is not missing math.

The blocker is:

- frozen-output insufficiency
- plus evaluator nondeterminism

## Next Honest Step

Before any harmonization rerun, the repo should first harden the evaluator:

1. fix and record an explicit evaluation seed
2. emit defended-style secondary metrics
3. persist score/probability artifacts needed for later review

Only after that should the lane decide whether to rerender:

- `baseline local63`
- frozen `SMP-LoRA local63`

## Verdict

- `secondary_metric_harmonization_audit = negative but useful`
- `post_hoc_harmonization_possible = no`
- `evaluator_can_be_extended = yes`
- `artifact_safe_rerender_today = no`
- `current_blocker = missing score artifacts and unseeded evaluation split`
- `gpu_release = none`
- `next_step = harden the local evaluator before any harmonization rerun`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this changes white-box execution discipline only.
