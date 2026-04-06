# 2026-04-07 White-Box Follow-Up: GSA End-to-End Execution Smoke

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-07 17:55:00 +08:00`
- `selected_mainline`: `2025 PoPETS White-box Membership Inference Attacks against Diffusion Models (GSA)`
- `current_state`: `end-to-end executable on CPU`
- `gpu_usage`: `not requested`
- `evidence_level`: `execution-smoke`

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

The result to foreground is execution, not toy attack strength.

Confirmed facts:

- four gradient artifacts were produced
- each split produced a `6 x 450` gradient tensor
- upstream `test_attack_accuracy.py` completed and wrote a target-side attack report
- raw toy metrics remain available in [attack-output.txt](runs/gsa-closed-loop-smoke-20260407-cpu/attack-output.txt)

Those raw numbers are intentionally not foregrounded here because this run still uses toy synthetic assets and should be read only as an execution proof.

## D. Interpretation

What this run proves:

- the upstream `GSA` DDPM gradient extractor works on this machine in CPU mode
- the upstream `xgboost` classifier stage also works on this machine
- the white-box line is no longer only `gradient-smoke`; it now has a minimal end-to-end executable path

What this run does **not** prove:

- paper-faithful white-box reproduction
- meaningful benchmark strength
- anything about real target/shadow checkpoints

This is still a toy synthetic-assets execution smoke.

## E. Remaining Blockers

- paper-aligned `target/shadow` checkpoints are still missing
- paper-aligned `member/non-member` splits are still missing
- the current result still sits in workspace scope; there is no shared `diffaudit` white-box adapter / CLI yet
- the tiny toy split triggers expected small-sample warnings in `sklearn`, so the raw toy numbers must not be read as publishable attack signal

## F. Shortest Next Step

1. Replace the toy buckets with paper-aligned `target-member`, `target-nonmember`, `shadow-member`, and `shadow-nonmember` assets.
2. Re-run gradient extraction against real checkpoints with `--resume_from_checkpoint latest`.
3. Keep the result in “end-to-end executable” language until those real assets exist.
