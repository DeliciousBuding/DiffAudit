# 2026-04-16 MoFit Launch-Budget Tightening Verdict

## Question

Can the `MoFit` lane now tighten the first-launch budget in code so the default script path is safe for a future admitted local CPU canary, without immediately spending real local runtime?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/tests/test_mofit_interface_canary.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_mofit_scaffold.py`
- `<DIFFAUDIT_ROOT>/Research/scripts/run_mofit_interface_canary.py`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-mofit-real-asset-canary-launch-gate-review.md`

## Verification

Failing test first:

- `conda run -n diffaudit-research python -m unittest tests.test_mofit_interface_canary`
- first result:
  - missing `apply_launch_profile`
  - missing `launch_profile` propagation in `run_canary`

Passing test after implementation:

- `conda run -n diffaudit-research python -m unittest tests.test_mofit_interface_canary`
- final result:
  - `Ran 3 tests ... OK`

Regression sweep:

- `conda run -n diffaudit-research python -m unittest tests.test_mofit_scaffold tests.test_mofit_interface_canary`
- final result:
  - `Ran 14 tests ... OK`

## What Landed

The script now additionally exposes:

- `apply_launch_profile(...)`

Current role:

- default `launch_profile = bounded-cpu-first`
- force first-launch budget down to:
  - `member_limit = 1`
  - `nonmember_limit = 1`
  - `surrogate_steps = 1`
  - `embedding_steps = 2`
  - `device = cpu`
- propagate the effective launch profile and tightened parameters into:
  - `run_canary(...)`
  - `summary.json`
  - test-visible orchestration behavior

## Verdict

- `launch_budget_tightening_verdict = positive`
- the repo no longer relies on an over-wide default first launch
- the next honest live step is now a fresh real local CPU canary, not another budget debate
- `gpu_release = none`

## Carry-Forward Rule

- keep `gpu_release = none`
- next step should launch exactly one fresh bounded local CPU canary under the tightened default profile

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
