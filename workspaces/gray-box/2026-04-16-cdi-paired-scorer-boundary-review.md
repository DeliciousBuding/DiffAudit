# 2026-04-16 CDI Paired-Scorer Boundary Review

## Question

Given the new `control-z-linear` paired scorer and the first repaired `2048` paired canary, should this scorer remain only a one-off internal canary, or is it now strong enough to become the default internal paired scorer for the current `CDI` lane?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-cdi-paired-feature-scorer-design.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/cdi-paired-canary-20260416-r1/audit_summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/cdi-paired-canary-20260416-r2-reverse/audit_summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-secmi-pia-2048-repaired-paired-surface-verdict.md`

## Review

### 1. The scorer is not a one-split fluke

Forward split (`r1`):

- `paired_t = 30.027926`
- `secmi_t = 29.637878`
- `pia_t = 27.141034`
- weights:
  - `secmi_stat = 0.526839`
  - `pia = 0.473161`

Reverse split (`r2-reverse`):

- `paired_t = 33.263280`
- `secmi_t = 34.234093`
- `pia_t = 29.172645`
- weights:
  - `secmi_stat = 0.515785`
  - `pia = 0.484215`

Interpretation:

- the scorer stays strongly same-directional under both half-split orientations
- the learned weights remain stable rather than collapsing onto one feature
- this is enough to reject the reading that the scorer only works on one arbitrary control/test ordering

### 2. But it is not a clean dominance scorer

- on the forward split, paired slightly exceeds `SecMI`
- on the reverse split, paired remains strong but does **not** beat `SecMI`

Interpretation:

- the scorer is useful and stable
- but current evidence does **not** justify claiming that paired scoring consistently dominates the stronger single feature
- so it should not yet become:
  - headline gray-box wording
  - external copyright-grade evidence
  - a replacement for reporting component statistics

### 3. The honest boundary

Current best reading:

- promote it from “single bounded canary only”
- but stop at “default internal paired scorer on the repaired `2048` shared surface”

Operational rule:

1. when running internal paired `CDI` checks on this surface, use `control-z-linear` as the default paired scorer
2. always report:
   - paired statistic
   - `SecMI` component statistic
   - `PIA` component statistic
3. do not describe the paired scorer as a benchmarked or externally validated superiority claim

## Verdict

- `cdi_paired_scorer_boundary_verdict = positive but constrained`
- the scorer is now strong enough to become the default internal paired scorer for the current `CDI` lane
- it is **not** strong enough to become a headline replacement for `SecMI` or `PIA`
- `gpu_release = none`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
