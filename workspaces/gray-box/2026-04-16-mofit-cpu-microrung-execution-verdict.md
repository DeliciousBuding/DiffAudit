# 2026-04-16 MoFit CPU Micro-Rung Execution Verdict

## Question

Can the frozen `2x2 / 2+4 / cpu` `MoFit` micro-rung actually execute end to end on the admitted local `SD1.5 + celeba_partial_target/checkpoint-25000` stack?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\tests\test_mofit_scaffold.py`
- `D:\Code\DiffAudit\Research\tests\test_mofit_interface_canary.py`
- `D:\Code\DiffAudit\Research\scripts\run_mofit_interface_canary.py`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\mofit-sd15-celeba-microrung-20260416-cpu-r2\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\mofit-sd15-celeba-microrung-20260416-cpu-r2\records.jsonl`

## Verification

Regression sweep before launch:

- `conda run -n diffaudit-research python -m unittest tests.test_mofit_scaffold tests.test_mofit_interface_canary`
- result:
  - `Ran 19 tests ... OK`

Fresh micro-rung launch:

- `conda run -n diffaudit-research python D:\Code\DiffAudit\Research\scripts\run_mofit_interface_canary.py --launch-profile cpu-micro-rung --run-root D:\Code\DiffAudit\Research\workspaces\gray-box\runs\mofit-sd15-celeba-microrung-20260416-cpu-r2 --member-dir D:\Code\DiffAudit\Research\workspaces\black-box\runs\clid-paper-align-asset-prep-20260415-r1\datasets\member --nonmember-dir D:\Code\DiffAudit\Research\workspaces\black-box\runs\clid-paper-align-asset-prep-20260415-r1\datasets\nonmember --model-dir D:\Code\DiffAudit\Download\shared\weights\stable-diffusion-v1-5 --lora-dir D:\\Code\\DiffAudit\\Download\\black-box\\supplementary\\recon-assets\\ndss-2025-blackbox-membership-inference-fine-tuned-diffusion-models\model-checkpoints\celeba_partial_target\checkpoint-25000 --blip-dir D:\Code\DiffAudit\Download\shared\weights\blip-image-captioning-large`
- final result:
  - `status = canary_executed`
  - `executed_member_count = 2`
  - `executed_nonmember_count = 2`

## What Landed

The first valid micro-rung now exists at:

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\mofit-sd15-celeba-microrung-20260416-cpu-r2`

Execution shape:

- `launch_profile = cpu-micro-rung`
- `member_limit = 2`
- `nonmember_limit = 2`
- `surrogate_steps = 2`
- `embedding_steps = 4`
- `device = cpu`

Execution note:

- an earlier `r1` attempt used the old profile semantics and did not represent the frozen rung
- that invalid attempt was discarded after the profile was corrected to exact `2x2 / 2+4 / cpu`

## Verdict

- `cpu_microrung_execution_verdict = positive but bounded`
- the frozen CPU micro-rung is now real execution evidence rather than design-only intent
- this closes the remaining “can the 2x2 rung run at all?” question
- `gpu_release = none`

## Carry-Forward Rule

- keep `gpu_release = none`
- next step should judge the score shape and cost of this rung before any larger CPU or GPU move

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

