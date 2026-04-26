# 2026-04-16 CDI Paired-Feature Scorer Design

## Question

With the repaired `PIA + SecMI` paired surface now available, what is the smallest honest paired `CDI` scorer the repo can implement and validate without overfitting or reopening GPU spend?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-cdi-feature-collection-surface-review.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-cdi-paired-feature-repromotion-review.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\secmi-pia-disagreement-20260416-r3\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-cifar10-runtime-mainline-20260416-gpu-2048-cdi-r1\scores.json`
- `D:\Code\DiffAudit\Research\scripts\run_cdi_internal_canary.py`

## Design

Frozen scorer:

- name: `control-z-linear`
- features:
  - `SecMI stat`
  - `PIA`
- fitting rule:
  - orient each feature into memberness direction first
  - use `P_ctrl + U_ctrl` to compute per-feature control mean / std
  - z-standardize each feature using only control statistics
  - assign linear weights from positive control-gap magnitude
  - apply the frozen linear scorer only on `P_test` and `U_test`

Why this is the smallest honest paired scorer:

1. it is control-fitted, not test-fitted
2. it is linear and auditable
3. it avoids introducing a learned classifier or opaque calibration layer
4. it uses only artifacts already landed on the repaired `2048` paired surface

## Bounded Validation

Run root:

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\cdi-paired-canary-20260416-r1`

Frozen split:

- `P_ctrl = first 1024`
- `P_test = last 1024`
- `U_ctrl = first 1024`
- `U_test = last 1024`

Observed results:

- `SecMI`:
  - `t = 29.637878`
- `PIA`:
  - `t = 27.141034`
- paired scorer:
  - `paired_t = 30.027926`
  - `paired_p = 1.279495359674726e-151`

Recovered control-fitted weights:

- `secmi_stat = 0.526839`
- `pia = 0.473161`

Interpretation:

- the paired scorer is not dominated by one feature
- and it modestly improves over the stronger single feature on this bounded internal canary

## Verdict

- `cdi_paired_feature_scorer_design_verdict = positive but bounded`
- the repo now has one honest paired `CDI` scorer design
- this scorer is suitable for internal paired-canary use on the repaired `2048` shared surface
- it should still be described as:
  - internal audit-shape paired scorer
  - not external copyright-grade evidence
- `gpu_release = none`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
