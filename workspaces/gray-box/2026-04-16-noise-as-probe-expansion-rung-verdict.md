# 2026-04-16 Noise-as-a-Probe Expansion Rung Verdict

## Question

After the first successful `1 + 1` interface canary and the fixed `8 / 8 / 8` policy, does the first bounded expansion rung keep the signal alive on the same local contract surface?

## Run Anchor

- `workspaces/gray-box/runs/noise-as-probe-expansion-rung-20260416-r1/summary.json`

## Contract

- target family:
  - `SD1.5 + celeba_partial_target/checkpoint-25000`
- split:
  - `8 members + 8 evaluation non-members + 8 calibration non-members`
- prompt source:
  - metadata text
- inversion steps:
  - `10`
- generation steps:
  - `10`
- metric:
  - `MSE`
- threshold policy:
  - calibration non-members only
  - `15th percentile`

## Result

The rung completed end to end.

Aggregate `MSE`:

- member mean:
  - `1068.7428`
- evaluation non-member mean:
  - `2031.9085`
- calibration non-member mean:
  - `2029.0161`

Threshold candidate from calibration non-members:

- `threshold = 1304.8905`

Threshold behavior on the evaluation rung:

- `accuracy = 0.75`
- `TPR = 0.75`
- `FPR = 0.25`

## Interpretation

This is enough to claim:

- the signal does not vanish immediately once the canary expands beyond `1 + 1`
- the simple percentile-style calibration policy is at least directionally usable on the local contract

This is **not** enough to claim:

- benchmark stability
- low-FPR quality
- release-grade gray-box challenger promotion

## Verdict

- `expansion_rung_verdict = positive`
- `current_verdict = positive but bounded`
- `gpu_release = none`
- `next_step = decide whether one disjoint repeat or threshold-hardening is the highest-value next bounded rung`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this is still a local bounded rung, not a project-level claim upgrade.
