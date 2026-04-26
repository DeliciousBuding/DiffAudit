# 2026-04-16 Noise-as-a-Probe Larger Rung Verdict

## Question

After `GB-12` threshold hardening, does a larger disjoint `16 / 16 / 16` rung keep the signal alive under the same local contract, and do the previously frozen conservative thresholds still work on the larger split?

## Run Anchor

- `workspaces/gray-box/runs/noise-as-probe-larger-rung-20260416-r3/summary.json`

## Contract

- target family:
  - `SD1.5 + celeba_partial_target/checkpoint-25000`
- split:
  - `16 members + 16 eval non-members + 16 calibration non-members`
- offsets:
  - `member = 16..31`
  - `calibration non-member = 32..47`
  - `eval non-member = 48..63`
- prompt source:
  - metadata text
- inversion steps:
  - `10`
- generation steps:
  - `10`
- metric:
  - `MSE`

Frozen threshold references carried in from `GB-12`:

- `frozen_r1 = 1304.8905`
- `frozen_pooled15 = 1308.7131`

## Result

The larger bounded rung completed end to end.

Aggregate `MSE`:

- member mean:
  - `961.4520`
- evaluation non-member mean:
  - `2203.2836`
- calibration non-member mean:
  - `2000.3564`

Self-calibrated threshold from the new rung:

- `threshold = 1326.5686`
- `accuracy = 0.90625`
- `TPR = 0.8125`
- `FPR = 0.0`

Frozen-threshold behavior on the same larger evaluation rung:

- `frozen_r1 = 1304.8905`
  - `accuracy = 0.90625`
  - `TPR = 0.8125`
  - `FPR = 0.0`
- `frozen_pooled15 = 1308.7131`
  - `accuracy = 0.90625`
  - `TPR = 0.8125`
  - `FPR = 0.0`

## Interpretation

This is enough to claim:

- the branch stays strongly same-directional on a larger disjoint bounded rung
- the `GB-12` conservative threshold band generalizes cleanly onto the larger split
- the current local `Noise as a Probe` line has now moved beyond mere canary survival:
  - it is still bounded
  - but it is no longer only a tiny-split curiosity

This is **not** enough to claim:

- release-grade gray-box challenger promotion
- paper-faithful benchmark parity
- that the branch should displace `PIA` or `TMIA-DM` in current project-level packaging

The correct reading is:

- stronger bounded challenger candidate
- still needs one same-scale disjoint repeat before any bigger narrative promotion

## Verdict

- `larger_rung_verdict = positive`
- `current_verdict = positive and strengthened but still bounded`
- `threshold_generalization = positive`
- `gpu_release = none`
- `next_step = run one disjoint repeat at the same 16 / 16 / 16 scale before any promotion discussion`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this materially strengthens the research branch, but it still does not change current top-level packaging.
