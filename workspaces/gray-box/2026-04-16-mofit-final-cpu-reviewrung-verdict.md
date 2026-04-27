# 2026-04-16 MoFit Final CPU Review-Rung Verdict

## Question

Can one final bounded CPU review rung materially improve `MoFit` score separation under the current local `SD1.5 + celeba_partial_target/checkpoint-25000` contract?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/tests/test_mofit_scaffold.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_mofit_interface_canary.py`
- `<DIFFAUDIT_ROOT>/Research/scripts/run_mofit_interface_canary.py`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/mofit-sd15-celeba-reviewrung-20260416-cpu-r2/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/mofit-sd15-celeba-reviewrung-20260416-cpu-r2/records.jsonl`

## Verification

Regression sweep before launch:

- `conda run -n diffaudit-research python -m unittest tests.test_mofit_scaffold tests.test_mofit_interface_canary`
- result:
  - `Ran 20 tests ... OK`

Fresh review-rung launch:

- `conda run -n diffaudit-research python <DIFFAUDIT_ROOT>/Research/scripts/run_mofit_interface_canary.py --launch-profile cpu-review-rung --run-root <DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/mofit-sd15-celeba-reviewrung-20260416-cpu-r2 --member-dir <DIFFAUDIT_ROOT>/Research/workspaces/black-box/runs/clid-paper-align-asset-prep-20260415-r1/datasets/member --nonmember-dir <DIFFAUDIT_ROOT>/Research/workspaces/black-box/runs/clid-paper-align-asset-prep-20260415-r1/datasets/nonmember --model-dir <DIFFAUDIT_ROOT>/Download/shared/weights/stable-diffusion-v1-5 --lora-dir <DIFFAUDIT_ROOT>/Download/black-box/supplementary/recon-assets/ndss-2025-blackbox-membership-inference-fine-tuned-diffusion-models/model-checkpoints/celeba_partial_target/checkpoint-25000 --blip-dir <DIFFAUDIT_ROOT>/Download/shared/weights/blip-image-captioning-large`
- final result:
  - `status = canary_executed`
  - `executed_member_count = 2`
  - `executed_nonmember_count = 2`

## What Landed

The final bounded CPU review rung now exists at:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/mofit-sd15-celeba-reviewrung-20260416-cpu-r2`

Execution shape:

- `launch_profile = cpu-review-rung`
- `member_limit = 2`
- `nonmember_limit = 2`
- `surrogate_steps = 3`
- `embedding_steps = 6`
- `device = cpu`

Observed score means:

- member mean: `-0.0039209`
- nonmember mean: `-0.0044675`
- gap: `+0.0005466`

Representative trace behavior:

- member embedding: `0.3253375 -> 0.3218594`
- nonmember embedding: `0.4098801 -> 0.4060381`

## Verdict

- `final_cpu_reviewrung_verdict = weak-positive but still bounded`
- the final bounded CPU rung preserves the same direction as the micro-rung
- the gap improves slightly, but remains too small to justify direct scale-up or GPU escalation
- `gpu_release = none`

## Carry-Forward Rule

- keep `gpu_release = none`
- next step should now decide whether the current `MoFit` contract should be placed on hold / no-go instead of spending more bounded runtime

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

