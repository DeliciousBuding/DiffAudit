# 2026-04-17 CDI Paired-Scorer Consumer Handoff Note

## Question

Now that the paired scorer contract is frozen, what should `Leader`, materials writers, and any future `Platform/Runtime` consumer read first from `audit_summary.json`, and what must they avoid over-claiming?

## Consumer Priority

Read the current paired `CDI` artifact in this order:

1. `contract`
2. `feature_mode`
3. `metrics`
4. `notes`
5. `analysis`

This order is intentionally asymmetric:

- `contract` tells higher layers what the run is allowed to mean
- `metrics` tell internal consumers what the run measured
- `analysis` only explains how the paired scorer was built and should stay diagnostic-first

## Consumer-Specific Rule

### `Leader` / summary-layer / materials draft

Consume first:

- `contract.version`
- `contract.feature_mode`
- `contract.paired_scorer_policy_effective`
- `contract.component_reporting_required`
- `contract.headline_use_allowed`
- `contract.external_evidence_allowed`
- `notes`

Safe wording:

- the paired `CDI` canary has a stable machine-readable internal contract
- paired mode is active only when the effective policy says so
- paired runs are required to keep reporting `paired + SecMI + PIA` together
- this artifact is internal-only and is not approved for headline or external-evidence use

Do not surface as summary/materials headline:

- `metrics.paired_t_statistic`
- `metrics.secmi_t_statistic`
- `metrics.pia_t_statistic`
- any claim that paired scoring "beats" `SecMI` in a stable project-level sense

### Future `Platform` / `Runtime` consumer

Hard-gate on:

- `contract.name`
- `contract.version`
- `contract.feature_mode`
- `contract.paired_scorer_policy_effective`
- `contract.component_reporting_required`
- `contract.headline_use_allowed`
- `contract.external_evidence_allowed`

If `contract.paired_scorer_policy_effective = none`:

- treat the artifact as component-only
- require `SecMI` component fields
- do not render paired-scorer-specific labels

If `contract.paired_scorer_policy_effective = control-z-linear`:

- require `paired + SecMI + PIA` metric triplet together
- allow internal UI or API labeling as `default internal paired scorer`
- still suppress any headline/public promotion path because both boundary flags remain `false`

Treat as diagnostic-only:

- `analysis.paired_scorer`
- `analysis.paired_scorer_details`
- scorer weights, centers, scales, and control gaps

### Research-side diagnostic consumer

May read:

- the full `metrics` block
- the full `analysis` block
- local artifact paths under `artifacts`

But even inside Research, the current artifact still does **not** justify:

- external evidence wording
- benchmark-style headline wording
- portability claims across datasets, models, or split contracts

## Safe Fields For Summary/Materials

Only these fields are safe to consume above the lane-local level without extra explanation:

- `contract.name`
- `contract.version`
- `contract.feature_mode`
- `contract.paired_scorer_policy_effective`
- `contract.component_reporting_required`
- `contract.headline_use_allowed`
- `contract.external_evidence_allowed`
- `notes`

## Fields That Must Not Be Overinterpreted

- `metrics.paired_t_statistic` is an internal canary strength signal, not a publishable gray-box headline
- `metrics.paired_t_statistic > metrics.secmi_t_statistic` does not prove a robust superior family; the boundary review already says the paired scorer does not stably dominate `SecMI`
- `analysis.paired_scorer_details.weights` are control-fitted local coefficients, not scientific importance weights
- `analysis.paired_scorer_details.features.*.center/scale/control_gap` are local fitting diagnostics, not deployment thresholds
- `duration_seconds` is irrelevant to research strength
- `artifacts.*` paths are reproducibility pointers, not consumption semantics

## Verification

Reviewed the contract-bearing paired artifact:

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\cdi-paired-canary-20260417-r3-contract\audit_summary.json`

Confirmed:

- `contract.paired_scorer_policy_effective = control-z-linear`
- `contract.component_reporting_required = true`
- `contract.headline_use_allowed = false`
- `contract.external_evidence_allowed = false`
- component metrics and paired metrics are all present together

## Verdict

- `cdi_paired_scorer_consumer_handoff_verdict = positive`
- higher-layer consumers now have an explicit read order and anti-overclaim rule
- this closes the last missing consumption layer around the current paired `CDI` scorer
- no new GPU question is justified by this handoff clarification

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `Leader/materials`: suggested only if they plan to ingest `CDI` paired artifacts directly
- `Platform/Runtime`: suggested only for future internal ingestion
- `competition_material_sync = none`
