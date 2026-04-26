# 2026-04-16 Black-Box Semantic-Aux Fusion Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `BB-2.2 / bounded fusion experiments`
- `method_family`: `semantic-auxiliary-classifier`
- `device`: `cpu`
- `decision`: `negative but useful`

## Question

Within the current `semantic-auxiliary-classifier` challenger, can a bounded score-fusion variant beat `mean_cos` strongly enough to justify reopening the scoring line as more than a calibration-layer refinement?

## Executed Evidence

Fusion run summary:

- `D:\Code\DiffAudit\Research\workspaces\black-box\runs\semantic-aux-fusion-20260416-r1\summary.json`

Comparator record sources:

- `D:\Code\DiffAudit\Research\workspaces\black-box\runs\semantic-aux-classifier-comparator-20260415-r1\outputs\records.json`
- `D:\Code\DiffAudit\Research\workspaces\black-box\runs\semantic-aux-classifier-comparator-20260416-r2\outputs\records.json`

Bounded candidates tested:

1. `mean_cos`
2. `max_cos`
3. `cosine_pair_zmean`
4. `cosine_ssim_triplet_zmean`
5. `cosine_ssim_rank_fusion`

## Metrics

### `16 / 16` comparator (`r1`)

- `mean_cos`:
  - `AUC = 0.945312`
  - `ASR = 0.875`
- best fusion candidate:
  - `cosine_pair_zmean`
  - `AUC = 0.921875`
  - `ASR = 0.875`
- conclusion:
  - no fusion candidate beat `mean_cos`

### `32 / 32` comparator (`r2`)

- `mean_cos`:
  - `AUC = 0.916992`
  - `ASR = 0.875`
- best fusion candidate:
  - `cosine_pair_zmean`
  - `AUC = 0.918945`
  - `ASR = 0.875`
  - `AUC gain vs mean_cos = 0.001953`
  - `Spearman vs mean_cos = 0.993132`
- conclusion:
  - the best bounded fusion only nudges the same ordering; it does not create a materially different score family

## Verdict

Current verdict:

- `negative but useful`

Reason:

1. bounded fusion does not beat `mean_cos` on the smaller comparator;
2. on the larger comparator, the best fusion gain is only `+0.001953 AUC`, which is far below a meaningful promotion bar;
3. the best fusion candidate is still almost rank-identical to `mean_cos` (`Spearman = 0.993132`);
4. therefore the current black-box semantic-aux line remains a `mean_cos`-led returned-image similarity signal, not a newly strengthened fusion-based challenger.

## Decision

Current decision:

- close `BB-2.2` as `negative but useful`
- keep `mean_cos` as the simplest semantic-aux score reference
- do not reopen fusion unless a new feature family or cross-method package appears

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: no direct change needed.
- Materials: if this challenger is mentioned, the honest wording is that bounded fusion was reviewed and did not materially improve the current `mean_cos`-led ordering.
