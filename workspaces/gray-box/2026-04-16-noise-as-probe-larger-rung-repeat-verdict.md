# 2026-04-16 Noise-as-a-Probe Larger Rung Repeat Verdict

## Question

After the strong first `16 / 16 / 16` bounded rung, does one disjoint repeat at the same scale keep the signal and the frozen conservative threshold behavior alive, or was `GB-13` still partly a lucky split?

## Run Anchor

- `workspaces/gray-box/runs/noise-as-probe-larger-rung-repeat-20260416-r4/summary.json`

## Contract

- target family:
  - `SD1.5 + celeba_partial_target/checkpoint-25000`
- split:
  - `16 members + 16 eval non-members + 16 calibration non-members`
- offsets:
  - `member = 32..47`
  - `calibration non-member = 64..79`
  - `eval non-member = 80..95`
- prompt source:
  - metadata text
- inversion steps:
  - `10`
- generation steps:
  - `10`
- metric:
  - `MSE`

Frozen threshold references carried in:

- `frozen_r1 = 1304.8905`
- `frozen_pooled15 = 1308.7131`
- `r3 self-threshold = 1326.5686`

## Result

The same-scale disjoint repeat completed end to end.

Aggregate `MSE`:

- member mean:
  - `916.0670`
- evaluation non-member mean:
  - `2064.2385`
- calibration non-member mean:
  - `2154.0927`

Self-calibrated threshold from the repeat rung:

- `threshold = 1522.8881`
- `accuracy = 0.9375`
- `TPR = 0.9375`
- `FPR = 0.0625`

Frozen-threshold behavior on the same repeat evaluation rung:

- `frozen_r1 = 1304.8905`
  - `accuracy = 0.9375`
  - `TPR = 0.875`
  - `FPR = 0.0`
- `frozen_pooled15 = 1308.7131`
  - `accuracy = 0.9688`
  - `TPR = 0.9375`
  - `FPR = 0.0`
- `r3 self-threshold = 1326.5686`
  - `accuracy = 0.9688`
  - `TPR = 0.9375`
  - `FPR = 0.0`

Combined across the two same-scale `16 / 16 / 16` rungs (`r3 + r4`):

- `frozen_pooled15 = 1308.7131`
  - `accuracy = 0.9375`
  - `TPR = 0.875`
  - `FPR = 0.0`

## Interpretation

This is enough to claim:

- the larger-rung result was not a one-split fluke
- the branch is now same-scale `repeat-positive` at `16 / 16 / 16`
- the conservative frozen threshold story remains cleaner than the new self-calibrated `r4` threshold
- the local `Noise as a Probe` line has become a credible bounded gray-box challenger candidate rather than only an exploratory side branch

This is **not** enough to claim:

- release-grade promotion
- that the branch should replace `PIA` or `TMIA-DM` in current packaged gray-box hierarchy
- paper-faithful benchmark parity

The next highest-value task is therefore CPU-side:

- decide the correct challenger boundary and packaging language
- do not spend another GPU rung before that boundary review

## Verdict

- `larger_rung_repeat_verdict = positive`
- `repeat_status = same-scale repeat-positive`
- `threshold_generalization = positive`
- `current_verdict = strengthened bounded challenger candidate`
- `gpu_release = none`
- `next_step = run a challenger-boundary review against current gray-box packaging before any further GPU expansion`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this changes internal research ranking pressure, but not yet top-level external packaging.
