# 2026-04-16 MoFit Real Target-Path Wiring Verdict

## Question

Can the `MoFit` lane now bridge its loss-contract helpers onto the real `SD1.5` target-model noise-prediction path shape, so future optimization loops can consume actual `UNet`-style outputs instead of toy-only differentiable closures?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/tests/test_mofit_scaffold.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/mofit_scaffold.py`
- `<DIFFAUDIT_ROOT>/Research/scripts/run_structural_memorization_smoke.py`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-mofit-latent-loss-contract-verdict.md`

## Verification

Failing test first:

- `conda run -n diffaudit-research python -m unittest tests.test_mofit_scaffold`
- first result:
  - missing `build_unet_noise_predictor`
  - missing `compute_guided_target_noise`

Passing test after implementation:

- `conda run -n diffaudit-research python -m unittest tests.test_mofit_scaffold`
- final result:
  - `Ran 9 tests ... OK`

## What Landed

The scaffold now additionally exposes:

- `build_unet_noise_predictor(...)`
- `compute_guided_target_noise(...)`

Current role:

- adapt `UNet(...).sample`-style outputs into the existing `predict_noise_fn(latent, timestep, embedding)` contract
- coerce timestep / dtype / device handling into a reusable bridge rather than leaving it buried in future scripts
- build guided target noise from real `cond/uncond` predictions so the existing `L_cond / L_uncond / mofit_score` helpers can consume target-model-derived inputs

## Verdict

- `real_target_path_wiring_verdict = positive but bounded`
- the `MoFit` lane now has a real target-path bridge at the helper layer
- this closes the gap between:
  - actual `UNet` call shape
  - target-noise construction
  - existing loss-contract helpers
- the lane still remains below smoke because caption bootstrap, per-sample record execution, and end-to-end optimization loops are not yet assembled into one real sample path

## Carry-Forward Rule

- keep `gpu_release = none`
- next step should wire the new helper bridge into:
  - caption bootstrap
  - one bounded sample-level execution loop
  - record append/finalize over a real target-family sample

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
