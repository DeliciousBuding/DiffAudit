# 2026-04-16 CDI Protocol / Asset Contract

## Question

Can the current repo support one honest first `CDI`-style gray-box collection-level audit lane, or is `CDI` still only a paper-side idea under the present local contracts?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\docs\paper-reports\gray-box\2025-cvpr-cdi-copyrighted-data-identification-diffusion-models-report.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-15-pia-vs-secmi-graybox-comparison.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-15-graybox-ranking-sensitive-disagreement-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\secmi-pia-disagreement-20260415-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\plan.md`
- `D:\Code\DiffAudit\Research\ROADMAP.md`

## Current Repo Surface

### What already exists

1. The repo already has two real gray-box signal producers on one shared local `CIFAR-10 DDPM` surface:
   - `PIA`
   - `SecMI`
2. The repo already has same-split comparison truth:
   - `PIA` remains defended headline
   - `SecMI` is an independent corroboration line
3. The repo already has machine-readable per-sample score artifacts on that shared surface:
   - `secmi_scores_1024.json`
   - `disagreement_analysis.json`

### What does not exist yet

1. No current `P / U / control / test` collection schema exists.
2. No current set-level scoring head or Welch-`t` significance pipeline exists.
3. No current artifact contract exists for:
   - collection membership labels
   - repeated resampling
   - audit-level `p`-value summaries
4. Current repo truth is therefore still below paper-faithful `CDI`.

## Honest First Local Contract

### Selected local surface

- `target_family = CIFAR-10 DDPM gray-box shared-score surface`
- do **not** start on `SD1.5` or `latent-diffusion` assets
- do **not** start from white-box-only `GM`-style features

### Selected initial signal source

- start from already-landed gray-box score surfaces
- first contract should allow:
  - `SecMI stat`
  - optional paired `PIA` score alignment on the same split
- first contract should not require a new GPU run

### Minimum artifact schema

The first honest `CDI` lane should emit at least:

1. `collections.json`
   - fixed `P_ctrl / U_ctrl / P_test / U_test` membership
2. `sample_scores.jsonl`
   - one row per sample with method score(s)
3. `audit_summary.json`
   - collection means
   - test statistic
   - `p`-value
   - resampling count
4. one human-readable verdict note

## No-Go Triggers

Freeze the lane immediately as `no-go` if any of the following holds:

1. current local artifacts cannot supply stable per-sample score tables on one shared split
2. the local `P / U` construction cannot be made distribution-honest enough even for a bounded internal audit canary
3. the first contract silently drifts from gray-box audit into white-box-only feature dependence

## Verdict

- `contract_verdict = positive but bounded`
- the repo is ready for a real `CDI` contract-first lane
- but only as:
  - `collection-level audit extension`
  - `CPU-only`
  - `paper-inspired, not paper-faithful`
- `gpu_release = none`

## Next Step

The next honest live task should be:

- `CDI feature / collection-surface review`

That task should answer:

1. which existing per-sample artifacts are reusable as `CDI` inputs
2. how to construct one bounded internal `P/U` split honestly
3. whether the first canary should be:
   - one-method (`SecMI` only), or
   - paired-method (`PIA + SecMI`)

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

