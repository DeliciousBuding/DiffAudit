# 2026-04-17 CDI Paired-Scorer Machine-Readable Contract Note

## Question

After freezing the default-run policy, what exact machine-readable fields should higher-layer consumers rely on when reading internal paired `CDI` canary outputs?

## Contract

Current summary contract is carried in `audit_summary.json` and should be consumed in this order:

1. `contract`
2. `feature_mode`
3. `metrics`
4. `analysis`

### Required top-level contract fields

- `contract.name`
- `contract.version`
- `contract.feature_mode`
- `contract.paired_scorer_policy_requested`
- `contract.paired_scorer_policy_effective`
- `contract.component_reporting_required`
- `contract.headline_use_allowed`
- `contract.external_evidence_allowed`

### Required metric fields

Always required:

- `metrics.secmi_t_statistic`
- `metrics.secmi_p_value`

Required when paired policy is effective:

- `metrics.pia_t_statistic`
- `metrics.pia_p_value`
- `metrics.paired_t_statistic`
- `metrics.paired_p_value`

### Consumption rule

- if `contract.paired_scorer_policy_effective = none`
  - treat the run as component-only
- if `contract.paired_scorer_policy_effective = control-z-linear`
  - treat the run as default internal paired scorer mode
  - but still honor:
    - `headline_use_allowed = false`
    - `external_evidence_allowed = false`

## Verification

This round re-emitted a policy-aligned paired canary artifact with the contract fields present:

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\cdi-paired-canary-20260417-r3-contract\audit_summary.json`

## Verdict

- `cdi_paired_scorer_machine_readable_contract_verdict = positive`
- higher-layer consumers now have one stable machine-readable contract for internal paired `CDI` runs
- this remains an internal execution/reporting contract, not a benchmark claim

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `Platform/Runtime`: optional consumer update if they later ingest gray-box `CDI` summaries
- `competition_material_sync = none`
