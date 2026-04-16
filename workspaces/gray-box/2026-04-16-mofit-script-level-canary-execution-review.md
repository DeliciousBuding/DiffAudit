# 2026-04-16 MoFit Script-Level Canary Execution Review

## Question

After `GB-29`, what is the smallest honest script-level path to mount the new `MoFit` sample helper onto real local assets and the current `SD1.5` target-family stack?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\scripts\run_mofit_interface_canary.py`
- `D:\Code\DiffAudit\Research\scripts\run_structural_memorization_smoke.py`
- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\mofit_scaffold.py`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\plan.md`

## What Already Exists

### 1. The current canary entry already owns the right CLI contract

`run_mofit_interface_canary.py` already owns:

- asset-path arguments
- bounded member/nonmember limits
- `guidance_scale`
- surrogate / embedding optimization hyperparameters

So the next script step should extend this entrypoint rather than create a second `MoFit` script.

### 2. The real local target-family substrate already exists elsewhere

`run_structural_memorization_smoke.py` already provides the exact script-level pieces `MoFit` still needs:

- `load_rows(...)`
- `caption_for_image(...)`
- `encode_prompt(...)`
- `encode_image(...)`
- `UNet`-backed conditional/unconditional prediction substrate

### 3. The helper-layer `MoFit` path is now complete enough

`mofit_scaffold.py` now owns:

- prompt resolution
- record append/finalize
- surrogate optimization
- embedding optimization
- real target-path helper bridge
- sample-level execution assembly

So the missing work is no longer helper design; it is script-level mounting.

## Decision

Selected next implementation surface:

- extend `run_mofit_interface_canary.py`

Required minimal additions:

1. reuse or inline the bounded `load_rows(...)` path for one-row member/nonmember selection
2. add a minimal `MoFit` runner that reuses:
   - caption bootstrap
   - prompt encoding
   - image-to-latent encoding
   - raw `UNet` forward as the `execute_mofit_sample(...)` target-path backend
3. keep execution bounded to canary scale:
   - CPU-first
   - one or very few rows
   - no GPU release

Rejected alternative:

- starting a second new script

Reason:

- it would duplicate the CLI contract that `run_mofit_interface_canary.py` already owns
- it would increase branch surface without adding new research truth

## Verdict

- `script_level_canary_review_verdict = positive but bounded`
- the next honest live task is now clear: mount the sample-level helper into the existing canary script
- the shortest path is to extend `run_mofit_interface_canary.py` with bounded row loading and a minimal local runner, reusing the structural-memorization substrate where possible

## Carry-Forward Rule

- keep `gpu_release = none`
- next task should implement:
  - bounded row loading
  - prompt / latent / embedding preparation
  - one real local canary execution path in `run_mofit_interface_canary.py`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
