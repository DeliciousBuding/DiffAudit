# 2026-04-16 Noise-as-a-Probe Threshold Hardening Verdict

## Question

After two repeat-positive bounded `8 / 8 / 8` rungs, is the current calibration-only threshold rule stable enough to keep as a bounded local contract, or is the threshold still too split-sensitive to trust even locally?

## Inputs Reviewed

- `workspaces/gray-box/2026-04-16-noise-as-probe-expansion-rung-verdict.md`
- `workspaces/gray-box/2026-04-16-noise-as-probe-expansion-repeat-verdict.md`
- `workspaces/gray-box/runs/noise-as-probe-expansion-rung-20260416-r1/summary.json`
- `workspaces/gray-box/runs/noise-as-probe-expansion-rung-20260416-r2/summary.json`
- `scripts/run_noise_as_probe_interface_canary.py`

## Threshold Rule Verified

The current threshold rule is exactly:

- calibration non-members only
- `threshold = np.percentile(calibration_scores, 15.0)`
- predict `member` when `MSE <= threshold`

This is an implementation-verified rule, not a note-only description.

## Cross-Run Read

Per-run thresholds from the two bounded rungs:

- `r1 threshold = 1304.8905`
- `r2 threshold = 1376.3332`
- pooled calibration `15th percentile = 1308.7131`

Cross-run behavior:

- `r1` self-threshold on `r1 eval`:
  - `accuracy = 0.75`
  - `TPR = 0.75`
  - `FPR = 0.25`
- `r2` self-threshold on `r2 eval`:
  - `accuracy = 0.75`
  - `TPR = 0.875`
  - `FPR = 0.375`
- `r1 threshold` reused on `r2 eval`:
  - `accuracy = 0.8125`
  - `TPR = 0.875`
  - `FPR = 0.25`
- pooled calibration `15th percentile` reused on `r2 eval`:
  - `accuracy = 0.8125`
  - `TPR = 0.875`
  - `FPR = 0.25`

Combined across both bounded rungs:

- fixed `r1 threshold = 1304.8905`:
  - `accuracy = 0.7812`
  - `TPR = 0.8125`
  - `FPR = 0.25`
- pooled `15th percentile = 1308.7131`:
  - `accuracy = 0.7812`
  - `TPR = 0.8125`
  - `FPR = 0.25`
- pooled `10th percentile = 1211.3918`:
  - `accuracy = 0.8125`
  - `TPR = 0.75`
  - `FPR = 0.125`

## Interpretation

This is enough to claim:

- threshold scale is not wildly unstable across the first two disjoint bounded rungs
- the current calibration-only low-percentile policy remains locally coherent
- a frozen conservative threshold around the current `1305-1309` band is honest enough for the next bounded rung

This is **not** enough to claim:

- release-grade threshold maturity
- stable low-FPR behavior
- that the current branch is ready for promoted gray-box challenger packaging
- that the pooled `10th percentile` variant is already strong enough to replace the current `15th percentile` rule

The `10th percentile` pooled threshold is interesting because it cuts combined `FPR` from `0.25` to `0.125`, but it also drops combined `TPR` from `0.8125` to `0.75`, and the evidence base is still only two small bounded rungs. That makes it a future comparison candidate, not a protocol rewrite.

## Verdict

- `threshold_hardening_verdict = positive but bounded`
- `local_threshold_rule = keep calibration-only percentile thresholding`
- `frozen_local_threshold_band = 1304.8905 .. 1308.7131`
- `release_thresholding = no-go for now`
- `gpu_release = none`
- `next_step = define one larger bounded rung that reuses a frozen conservative threshold band before any stronger claim`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this changes local research truth and next-step discipline, but it does not yet change the public project-level story.
