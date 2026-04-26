# 2026-04-16 Noise-as-a-Probe Interface Canary Verdict

## Question

Can the first local `Noise as a Probe` interface canary actually run end to end on the selected `SD1.5 + celeba_partial_target/checkpoint-25000` contract surface?

## Run Anchor

- `workspaces/gray-box/runs/noise-as-probe-interface-canary-20260416-r1/summary.json`

## Contract

- target family:
  - `SD1.5 + celeba_partial_target/checkpoint-25000`
- split:
  - `1 member + 1 non-member`
- prompt source:
  - metadata text
- inversion steps:
  - `10`
- generation steps:
  - `10`
- metric:
  - `MSE`

## Result

The canary completed end to end and emitted:

1. query image
2. replay image
3. inverted latent
4. `summary.json`
5. `records.json`

Observed first-pair distances:

- member:
  - `MSE = 1180.3894`
  - `SSIM = 0.641248`
- non-member:
  - `MSE = 1850.7910`
  - `SSIM = 0.596142`

## Interpretation

This is enough to claim:

- the local three-stage path is real:
  - base-model inversion
  - target-model replay from injected latent
  - final distance scoring

This is **not** enough to claim:

- threshold quality
- calibration quality
- benchmark parity
- promotion into a released gray-box attack line

## Verdict

- `interface_canary_verdict = positive`
- `current_verdict = positive but bounded`
- `gpu_release = none`
- `next_step = freeze a calibration / expansion policy before any larger run`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- the canary proves local execution plumbing, but does not yet change project-level claims.
