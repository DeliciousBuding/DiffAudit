# 2026-04-16 MoFit Latent Loss Contract Verdict

## Question

Can the `MoFit` lane now express its real latent-path loss contract in code, so that future surrogate and fitted-embedding loops are optimizing the same `L_cond / L_uncond / mofit_score` structure the lane intends to report?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\tests\test_mofit_scaffold.py`
- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\mofit_scaffold.py`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-mofit-optimization-helper-verdict.md`

## Verification

Failing test first:

- `conda run -n diffaudit-research python -m unittest tests.test_mofit_scaffold`
- first result:
  - missing `compute_mofit_loss_terms`
  - missing `build_surrogate_loss_fn`
  - missing `build_embedding_loss_fn`

Passing test after implementation:

- `conda run -n diffaudit-research python -m unittest tests.test_mofit_scaffold`
- final result:
  - `Ran 7 tests ... OK`

## What Landed

The scaffold now exposes:

- `compute_mofit_loss_terms(...)`
- `build_surrogate_loss_fn(...)`
- `build_embedding_loss_fn(...)`

Current role:

- they define the real loss contract the lane will optimize later
- they are still verified only on toy differentiable predictors
- they now align optimization helpers with the exact score fields the lane writes out

## Verdict

- `contract_verdict = positive but bounded`
- the `MoFit` lane now has a real latent-path loss contract in code
- this removes the remaining schema/optimization mismatch between:
  - helper loops
  - record fields
  - final score semantics
- the lane still remains below smoke because the contract is not yet wired into the actual SD1.5 target-model path

## Carry-Forward Rule

- keep `gpu_release = none`
- next step should wire the new contract into:
  - the real target-family latent path
  - caption bootstrap
  - actual `UNet` noise-prediction calls

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
