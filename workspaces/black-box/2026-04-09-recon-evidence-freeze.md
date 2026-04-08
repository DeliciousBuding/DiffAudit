# 2026-04-09 Black-Box Follow-Up: Recon Evidence Freeze

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-09 00:20:00 +08:00`
- `selected_mainline`: `recon`
- `current_state`: `black-box evidence wording frozen`
- `evidence_level`: `runtime-mainline`

## Frozen Wording

Main evidence:

- `recon DDIM public-100 step30`
- summary:
  - `experiments/recon-runtime-mainline-ddim-public-100-step30/summary.json`
- metrics:
  - `auc = 0.849`
  - `asr = 0.51`
  - `tpr@1%fpr = 1.0`

Best single-metric reference:

- `recon DDIM public-50 step10`
- summary:
  - `experiments/recon-runtime-mainline-ddim-public-50-step10/summary.json`
- metric:
  - `auc = 0.866`

Secondary black-box track:

- `variation / Towards`
- status:
  - `formal local secondary track`
  - `blocked real-API assets`

Method boundary reminder:

- newly archived `TMIA-DM` does **not** enter this hierarchy
- `TMIA-DM` is currently a gray-box candidate paper because it depends on temporal noise / gradient information

## Interpretation

- `public-100 step30` is the main evidence because its artifact chain is the current most complete and most defensible.
- `public-50 step10` is not the main evidence; it is only the current best single AUC reference.
- `variation` remains useful in the repo, but it must not be overstated as a runnable real black-box line until a real query image root exists.
- `TMIA-DM` may be useful for literature comparison, but it does not change the current black-box execution ordering.
