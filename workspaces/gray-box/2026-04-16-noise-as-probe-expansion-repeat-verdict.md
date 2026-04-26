# 2026-04-16 Noise-as-a-Probe Expansion Repeat Verdict

## Question

Does a disjoint repeat of the first `8 / 8 / 8` `Noise as a Probe` rung keep the same direction, or was the first rung just a lucky split?

## Run Anchor

- `workspaces/gray-box/runs/noise-as-probe-expansion-rung-20260416-r2/summary.json`

## Contract

- target family:
  - `SD1.5 + celeba_partial_target/checkpoint-25000`
- split:
  - `8 members + 8 eval non-members + 8 calibration non-members`
- offsets:
  - `member = 8..15`
  - `eval non-member = 24..31`
  - `calibration non-member = 16..23`
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

The disjoint repeat completed end to end.

Aggregate `MSE`:

- member mean:
  - `839.3005`
- evaluation non-member mean:
  - `1528.4397`
- calibration non-member mean:
  - `2170.1746`

Threshold candidate from calibration non-members:

- `threshold = 1376.3332`

Threshold behavior on the evaluation rung:

- `accuracy = 0.75`
- `TPR = 0.875`
- `FPR = 0.375`

## Interpretation

This is enough to claim:

- the first expansion result was not a one-split fluke
- the direction remains stable across a disjoint bounded repeat:
  - member mean `MSE` still stays below non-member mean `MSE`
  - simple percentile-style calibration still produces non-random separation

This is **not** enough to claim:

- low-FPR maturity
- release-grade thresholding
- promoted gray-box challenger status

## Verdict

- `expansion_repeat_verdict = positive`
- `current_verdict = positive but bounded`
- `repeat_status = repeat-positive`
- `gpu_release = none`
- `next_step = choose between threshold-hardening and one larger bounded rung`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- the branch is stronger than a one-off canary now, but still below release-grade evidence.
