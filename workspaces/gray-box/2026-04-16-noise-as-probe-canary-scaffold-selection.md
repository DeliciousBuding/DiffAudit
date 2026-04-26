# 2026-04-16 Noise-as-a-Probe Canary Scaffold Selection

## Question

For the first `Noise as a Probe` interface canary, should the repo extend an existing script or add a dedicated canary script?

## Inputs Reviewed

- `workspaces/gray-box/2026-04-16-noise-as-probe-implementation-surface-review.md`
- `scripts/run_blackbox_semantic_aux_probe.py`
- `scripts/run_structural_memorization_smoke.py`
- `scripts/prepare_clid_local_bridge.py`

## Options

### Option A: extend `run_blackbox_semantic_aux_probe.py`

- Pros:
  - already has SD1.5 + LoRA loading
  - already has prompt fallback and image export
- Cons:
  - centered on prompt-conditioned return similarity
  - no latent inversion or custom-noise interface
  - would mix black-box semantic-aux logic with gray-box custom-noise logic

### Option B: extend `run_structural_memorization_smoke.py`

- Pros:
  - already has VAE encode/decode
  - already has `DDIMScheduler` and direct latent stepping
- Cons:
  - centered on structure-preservation corruption, not inversion + injected-noise replay
  - existing scoring/output semantics are specialized to another family

### Option C: add a dedicated `Noise as a Probe` canary script

- Pros:
  - keeps the new family's contract explicit
  - can reuse selected helpers from both existing scripts without semantic drift
  - produces a clean canary-specific output schema
- Cons:
  - small upfront duplication

## Selection

- `selected_option = dedicated canary script`

Recommended script shape:

- `scripts/run_noise_as_probe_interface_canary.py`

## Why This Wins

1. The family is now different enough that reusing an old script as the main surface would hide the real contract.
2. Existing scripts should be treated as helper donors:
   - prompt / image helper pieces from `semantic_aux`
   - latent / scheduler helper pieces from `structural_memorization`
3. A dedicated script keeps the output schema aligned with the canary:
   - source sample
   - prompt source
   - inversion artifact
   - generated replay artifact
   - distance metrics

## Verdict

- `scaffold_selection_verdict = positive`
- `selected_option = dedicated canary script`
- `next_step = implement the dedicated canary scaffold with reusable helper extraction only when helpful`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this is implementation-boundary truth only.
