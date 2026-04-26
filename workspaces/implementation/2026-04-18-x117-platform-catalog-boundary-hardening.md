# 2026-04-18 X-117 Platform Catalog Boundary Hardening

## Question

Does the current `Platform` public `catalog.json` overstate admitted `Research` truth, and if so, what is the smallest honest fix?

## Inputs Reviewed

- `D:\Code\DiffAudit\Platform\apps\api-go\data\public\catalog.json`
- `D:\Code\DiffAudit\Research\docs\admitted-results-summary.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-18-platform-intake-from-research.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-18-x115-research-runtime-platform-handoff-after-x114.md`

## What Was Found

The public snapshot `catalog.json` still contained consumer-facing copy that was stronger than current admitted research truth:

1. black-box `recon` copy jumped from admitted risk evidence to prescriptive defense wording
2. gray-box `PIA` copy described the defense effect without the required `provisional / not validated privacy protection` boundary
3. white-box `GSA/W-1` copy overstated the defended result as if risk were fully eliminated

## Actual Read

Current honest platform-facing interpretation should stay within these boundaries:

1. black-box `recon` = admitted risk evidence on a controlled/public-subset protocol, not a generalized exploit claim
2. gray-box `PIA + stochastic-dropout` = measurable but limited reduction under the current admitted protocol, not validated privacy protection
3. white-box `GSA/W-1` = strong mitigation signal under the present admitted comparison setup, not benchmark-complete and not global risk elimination

## Verdict

- `x117_platform_catalog_boundary_hardening_verdict = positive`

More precise reading:

1. current issue sits in consumer copy, not in admitted metrics
2. the smallest honest fix is to harden snapshot wording only
3. no Runtime endpoint, runner capability, or schema change is required

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `Platform handoff = snapshot copy hardening only`
- `Runtime handoff = none`
- `competition-material sync decision = optional wording-only downstream refresh`

## Canonical Evidence Anchor

- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-18-x117-platform-catalog-boundary-hardening.md`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Platform/apps/api-go/data/public/catalog.json`: update required
- `Runtime/Platform schema change`: not required
