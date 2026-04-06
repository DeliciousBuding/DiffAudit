# 2026-04-07 White-Box Follow-Up: GSA Closed-Loop Smoke

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-07 17:55:00 +08:00`
- `selected_mainline`: `2025 PoPETS White-box Membership Inference Attacks against Diffusion Models (GSA)`
- `current_state`: `closed-loop-smoke on CPU`
- `gpu_usage`: `not requested`
- `evidence_level`: `closed-loop-smoke`

## A. What Changed Before Running

- `xgboost==1.7.3` was added to [environment.yml](../../environment.yml) and installed into `diffaudit-research`.
- No `Platform` / `Services` / `LocalOps` code was touched.
- No GPU was requested.

## B. What Was Executed

This phase used a one-off CPU orchestration to stay inside the existing upstream GSA scripts:

1. Create four toy `imagefolder` buckets under [runs/gsa-closed-loop-smoke-20260407-cpu](runs/gsa-closed-loop-smoke-20260407-cpu).
2. Run upstream [gen_l2_gradients_DDPM.py](external/GSA/DDPM/gen_l2_gradients_DDPM.py) four times to generate:
   - `target_member-gradients.pt`
   - `target_non_member-gradients.pt`
   - `shadow_member-gradients.pt`
   - `shadow_non_member-gradients.pt`
3. Run upstream [test_attack_accuracy.py](external/GSA/test_attack_accuracy.py) once to train the XGBoost attack model and evaluate the target split.

Primary artifacts:

- [summary.json](runs/gsa-closed-loop-smoke-20260407-cpu/summary.json)
- [attack-output.txt](runs/gsa-closed-loop-smoke-20260407-cpu/attack-output.txt)

## C. Result

Target-side closed-loop metrics from [attack-output.txt](runs/gsa-closed-loop-smoke-20260407-cpu/attack-output.txt):

- `accuracy = 0.75`
- `roc_auc = 0.75`
- `tpr@1%fpr = 0.0`
- `tpr@0.1%fpr = 0.0`

Gradient artifact summary from [summary.json](runs/gsa-closed-loop-smoke-20260407-cpu/summary.json):

- each split produced a `6 x 450` gradient tensor
- `target_member` mean/std: `0.0592 / 0.2229`
- `target_non_member` mean/std: `0.0648 / 0.2492`
- `shadow_member` mean/std: `0.0577 / 0.2180`
- `shadow_non_member` mean/std: `0.0630 / 0.2438`

## D. Interpretation

What this run proves:

- the upstream `GSA` DDPM gradient extractor works on this machine in CPU mode
- the upstream `xgboost` classifier stage also works on this machine
- the white-box line is no longer only `gradient-smoke`; it now has a minimal `closed-loop-smoke`

What this run does **not** prove:

- paper-faithful white-box reproduction
- meaningful benchmark strength
- anything about real target/shadow checkpoints

This is still a toy synthetic-assets closed loop.

## E. Remaining Blockers

- paper-aligned `target/shadow` checkpoints are still missing
- paper-aligned `member/non-member` splits are still missing
- the current result still sits in workspace scope; there is no shared `diffaudit` white-box adapter / CLI yet
- the tiny toy split triggers expected small-sample warnings in `sklearn`, so these numbers are not publishable evidence

## F. Shortest Next Step

1. Replace the toy buckets with paper-aligned `target-member`, `target-nonmember`, `shadow-member`, and `shadow-nonmember` assets.
2. Re-run gradient extraction against real checkpoints with `--resume_from_checkpoint latest`.
3. Keep the result in `closed-loop-smoke` language until those real assets exist.
