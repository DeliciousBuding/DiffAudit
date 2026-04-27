# 2026-04-16 MoFit Fresh Real-Asset Canary Verdict

## Question

After the helper, script, and launch-budget work landed, can the `MoFit` lane now execute one fresh bounded local CPU canary on the admitted `SD1.5 + celeba_partial_target/checkpoint-25000` stack?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/scripts/run_mofit_interface_canary.py`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/mofit-sd15-celeba-canary-20260416-cpu-r4/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/mofit-sd15-celeba-canary-20260416-cpu-r4/records.jsonl`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-mofit-launch-budget-tightening-verdict.md`

## Verification

Regression sweep before launch:

- `conda run -n diffaudit-research python -m unittest tests.test_mofit_scaffold tests.test_mofit_interface_canary`
- result:
  - `Ran 17 tests ... OK`

Fresh real-asset canary launch:

- `conda run -n diffaudit-research python <DIFFAUDIT_ROOT>/Research/scripts/run_mofit_interface_canary.py --run-root <DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/mofit-sd15-celeba-canary-20260416-cpu-r4 --member-dir <DIFFAUDIT_ROOT>/Research/workspaces/black-box/runs/clid-paper-align-asset-prep-20260415-r1/datasets/member --nonmember-dir <DIFFAUDIT_ROOT>/Research/workspaces/black-box/runs/clid-paper-align-asset-prep-20260415-r1/datasets/nonmember --model-dir <DIFFAUDIT_ROOT>/Download/shared/weights/stable-diffusion-v1-5 --lora-dir <DIFFAUDIT_ROOT>/Download/black-box/supplementary/recon-assets/ndss-2025-blackbox-membership-inference-fine-tuned-diffusion-models/model-checkpoints/celeba_partial_target/checkpoint-25000 --blip-dir <DIFFAUDIT_ROOT>/Download/shared/weights/blip-image-captioning-large`
- final result:
  - `status = canary_executed`
  - `executed_member_count = 1`
  - `executed_nonmember_count = 1`

## What Landed

The first fresh real local CPU canary now exists at:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/mofit-sd15-celeba-canary-20260416-cpu-r4`

Execution shape:

- `launch_profile = bounded-cpu-first`
- `member_limit = 1`
- `nonmember_limit = 1`
- `surrogate_steps = 1`
- `embedding_steps = 2`
- `device = cpu`

Observed per-sample results:

- member:
  - `l_cond = 0.3830916`
  - `l_uncond = 0.3834537`
  - `mofit_score = -0.0003621`
- nonmember:
  - `l_cond = 0.4745996`
  - `l_uncond = 0.4752362`
  - `mofit_score = -0.0006366`

## Verdict

- `fresh_real_asset_canary_verdict = positive but bounded`
- the `MoFit` lane now has a fresh admitted local CPU canary execution, not just test-backed orchestration
- this closes the remaining “can it run at all on the real local stack?” question
- the resulting score gap is still tiny and not yet enough to justify direct scale-up

## Carry-Forward Rule

- keep `gpu_release = none`
- next step should review whether the current score/traces justify:
  - a slightly larger CPU micro-rung, or
  - an execution-side no-go / reformulation before more budget is spent

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

