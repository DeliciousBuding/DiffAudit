# 2026-04-16 CDI Paired-Scorer Default-Run Policy Note

## Question

Now that the `CDI` paired scorer boundary and summary wording are both frozen, what exact default-run policy should later sessions follow so the scorer is used consistently without overclaiming?

## Policy

### 1. Default policy

Use:

- `--paired-scorer auto`

Meaning:

- if only `SecMI` inputs are present:
  - stay on `SecMI-only`
- if paired inputs are present:
  - enable `control-z-linear` automatically

This makes the current default internal paired scorer the default execution behavior on paired-capable internal `CDI` runs, without changing one-method canaries.

### 2. Explicit override rules

- `--paired-scorer none`
  - use for component-only replay, ablation, or first-canary historical reproduction
- `--paired-scorer control-z-linear`
  - use when a session needs to force the paired scorer explicitly for audit clarity or reproducibility

### 3. Reporting rule

Whenever paired scoring is enabled, always report together:

1. paired statistic
2. `SecMI` component statistic
3. `PIA` component statistic

Reason:

- the boundary review already showed the paired scorer is useful and stable
- but it does not consistently dominate `SecMI`
- so dropping component statistics would overclaim

### 4. Boundary rule

Do not describe default paired runs as:

- headline scorer
- benchmarked superiority claim
- external copyright-grade evidence

Current honest label remains:

- `default internal paired scorer on the repaired 2048 surface`

## Verdict

- `cdi_paired_scorer_default_run_policy_verdict = positive`
- the repo now has a short executable default-run policy for later sessions
- the recommended default is:
  - `auto`
- `gpu_release = none`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
