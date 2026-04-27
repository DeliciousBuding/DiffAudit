# 2026-04-16 CDI Feature / Collection-Surface Review

## Question

Given the current local gray-box artifacts, what is the smallest honest `CDI` feature / collection surface the repo can already support, and should the first bounded canary be `SecMI-only` or paired `PIA + SecMI`?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-cdi-protocol-asset-contract.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-15-pia-vs-secmi-graybox-comparison.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-15-graybox-ranking-sensitive-disagreement-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260415-gpu-1024-adaptive/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260415-gpu-1024-adaptive/scores.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/secmi-pia-disagreement-20260415-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/secmi-pia-disagreement-20260415-r1/outputs/secmi_scores_1024.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/secmi-pia-disagreement-20260415-r1/outputs/disagreement_analysis.json`
- `<DIFFAUDIT_ROOT>/Research/docs/paper-reports/gray-box/2025-cvpr-cdi-copyrighted-data-identification-diffusion-models-report.md`

## Current Surface Review

### Reusable score artifacts

1. `PIA 1024 adaptive` already exposes:
   - `member_scores`
   - `nonmember_scores`
   - `member_indices`
   - `nonmember_indices`
2. `SecMI disagreement` already exposes:
   - `member_scores`
   - `nonmember_scores`
   on the same bounded `1024 / 1024` comparison surface
3. `disagreement_analysis.json` is not itself a CDI feature table, but it is strong alignment evidence:
   - `Spearman = 0.907581`
   - `disagreement_rate = 0.122559`

### What is honest now

- The repo already has enough to build one bounded internal `CDI` canary on a same-split local `CIFAR-10 DDPM` surface.
- But that canary is still an **internal audit-shape check**, not a copyright-grade external claim, because:
  - current `P` and `U` can only be constructed from member/non-member subsets already inside the audit stack
  - no real public-suspect vs private-control collection surface exists yet

## Frozen First Collection Contract

### Local collection construction

The first bounded internal `CDI` canary should use the current `1024 / 1024` same-split surface and freeze:

- `P = member subset`
- `U = nonmember subset`
- deterministic equal-size split:
  - `P_ctrl = first 512`
  - `P_test = last 512`
  - `U_ctrl = first 512`
  - `U_test = last 512`

Reason:

- it preserves equal `|P| = |U|`
- it stays within already executed local artifacts
- it is sufficient for one bounded `control -> test` audit-shape canary before any heavier resampling loop

### First feature choice

The first bounded canary should be:

- `SecMI stat only`

Why:

1. it is the cleanest exported per-sample score table already frozen for this surface
2. it avoids making the very first `CDI` canary depend on multi-method feature alignment assumptions
3. it keeps the first verdict focused on:
   - collection schema
   - test-statistic path
   - `p`-value emission

### Paired-method status

- paired `PIA + SecMI` is **allowed as follow-up**, not as the first canary default
- current `PIA 1024` scores are already reusable enough for later feature stacking
- but the repo should first prove that a one-method `CDI` canary can emit:
  - `collections.json`
  - `sample_scores.jsonl`
  - `audit_summary.json`
  without hiding behind multi-feature complexity

## GPU Follow-Up Decision

If GPU budget is used next for the `CDI` lane, the highest-value GPU task is:

- widen the reusable `PIA` shared-score surface beyond the current `1024 / 1024` rung

Reason:

- it directly improves the later paired-method `CDI` follow-up
- it does not reopen a stale standalone gray-box family
- it keeps GPU spend attached to the current live lane instead of inventing a separate branch

## Verdict

- `feature_collection_review_verdict = positive but bounded`
- the repo already supports one honest bounded internal `CDI` canary
- the frozen first canary shape is:
  - `same-split CIFAR-10 DDPM`
  - deterministic `512/512` control-test partition inside the existing `1024 / 1024` surface
  - `SecMI stat only`
- paired `PIA + SecMI` should be treated as the next extension, not the first default
- `gpu_release = none for CDI canary itself`
- `next_gpu_candidate = larger PIA shared-score surface refresh for CDI paired follow-up`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

