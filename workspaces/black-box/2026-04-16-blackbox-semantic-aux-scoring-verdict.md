# 2026-04-16 Black-Box Semantic-Aux Scoring Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `BB-2 / scoring and calibration upgrades`
- `method_family`: `semantic-auxiliary-classifier`
- `device`: `cpu`
- `decision`: `negative but useful`

## Question

Within the current `semantic-auxiliary-classifier` challenger, does the multi-feature logistic score provide a genuinely stronger ranking signal than the simplest single returned-image feature, or is it mostly a calibration layer without new ordering power?

## Executed Evidence

Comparator summaries:

- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/runs/semantic-aux-classifier-comparator-20260415-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/runs/semantic-aux-classifier-comparator-20260416-r2/summary.json`

Raw feature records:

- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/runs/semantic-aux-classifier-comparator-20260415-r1/outputs/records.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/runs/semantic-aux-classifier-comparator-20260416-r2/outputs/records.json`

## Metrics

### `16 / 16` comparator (`r1`)

- logistic multi-feature score:
  - `AUC = 0.910156`
- single-feature readouts:
  - `mean_cos = 0.945312`
  - `max_cos = 0.90625`
  - `mean_ssim = 0.640625`
  - `max_ssim = 0.605469`
- rank agreement:
  - `Spearman(logistic, mean_cos) = 0.973607`

### `32 / 32` comparator (`r2`)

- logistic multi-feature score:
  - `AUC = 0.90625`
- single-feature readouts:
  - `mean_cos = 0.916992`
  - `max_cos = 0.90625`
  - `mean_ssim = 0.629883`
  - `max_ssim = 0.62793`
- rank agreement:
  - `Spearman(logistic, mean_cos) = 0.978709`

## Verdict

Current verdict:

- `negative but useful`

Reason:

1. on both bounded comparator runs, the current multi-feature logistic score does **not** beat the simplest single feature `mean_cos`;
2. the logistic score is almost rank-equivalent to `mean_cos` (`Spearman ≈ 0.97–0.98`), so it is not opening a materially different ordering signal;
3. this means the current semantic-aux challenger is best understood as a returned-image semantic similarity line whose main readout is already visible in `mean_cos`;
4. the result is still useful because it tightens the method story and prevents the repo from overclaiming “multi-score challenger” value where there is mostly threshold/calibration behavior.

## Decision

Current decision:

- `do not promote current multi-feature logistic as a stronger challenger`
- `treat mean_cos as the current simplest black-box semantic-aux score reference`
- `only reopen BB-2 with a new feature family or a same-protocol cross-method score package`

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: no direct change needed.
- Materials: if the semantic-aux challenger is mentioned, the honest wording is that its current strength is already mostly captured by returned-image semantic similarity (`mean_cos`), not by a richer calibration stack.
