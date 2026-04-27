# 2026-04-16 MoFit Script-Level Canary Implementation Verdict

## Question

Can the existing `run_mofit_interface_canary.py` entry now execute a bounded script-level `MoFit` canary path instead of stopping at scaffold initialization only?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/tests/test_mofit_interface_canary.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_mofit_scaffold.py`
- `<DIFFAUDIT_ROOT>/Research/scripts/run_mofit_interface_canary.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/mofit_scaffold.py`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-mofit-script-level-canary-execution-review.md`

## Verification

Failing test first:

- `conda run -n diffaudit-research python -m unittest tests.test_mofit_interface_canary`
- first result:
  - missing `load_rows`
  - missing `run_canary`

Passing tests after implementation:

- `conda run -n diffaudit-research python -m unittest tests.test_mofit_interface_canary`
- final result:
  - `Ran 2 tests ... OK`

Regression sweep:

- `conda run -n diffaudit-research python -m unittest tests.test_mofit_scaffold tests.test_mofit_interface_canary`
- final result:
  - `Ran 13 tests ... OK`

## What Landed

The script now additionally exposes:

- `load_rows(...)`
- `MoFitCanaryRunner.execute_row(...)`
- `run_split(...)`
- `run_canary(...)`

Current role:

- perform bounded row selection with `offset/limit`
- mount the existing helper-layer `MoFit` path into the current canary script
- reuse the local structural-memorization substrate for:
  - caption bootstrap
  - prompt encoding
  - image-to-latent encoding
  - `UNet` target-path calls
- update `summary.json` from `scaffold_only` to `canary_executed`

## Verdict

- `script_level_canary_implementation_verdict = positive but bounded`
- `run_mofit_interface_canary.py` is no longer scaffold-only code
- the script now has a real bounded execution path at the orchestration layer
- the current evidence is still unit-test-backed rather than a fresh real-asset launch on the admitted local stack

## Carry-Forward Rule

- keep `gpu_release = none`
- next step should decide whether to:
  - launch one real local CPU canary with minimal `member/nonmember` counts, or
  - first tighten launch defaults to reduce unnecessary CPU waste

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
