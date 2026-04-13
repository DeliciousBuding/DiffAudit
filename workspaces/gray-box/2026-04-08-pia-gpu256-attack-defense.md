# 2026-04-08 Gray-Box Follow-Up: PIA GPU256 Attack-Defense Pair

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-08 21:49:17 +08:00`
- `selected_mainline`: `PIA`
- `current_state`: `single-GPU baseline and stochastic-dropout defense pair completed at 256 samples`
- `gpu_usage`: `single GPU, serial`
- `evidence_level`: `runtime-mainline`

## A. Commands Run

Baseline:

```powershell
conda run -n diffaudit-research python -m diffaudit run-pia-runtime-mainline `
  --config tmp/configs/pia-cifar10-graybox-assets.local.yaml `
  --workspace workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-256 `
  --repo-root external/PIA `
  --member-split-root external/PIA/DDPM `
  --device cuda:0 `
  --max-samples 256 `
  --batch-size 8 `
  --provenance-status source-retained-unverified
```

Defense:

```powershell
conda run -n diffaudit-research python -m diffaudit run-pia-runtime-mainline `
  --config tmp/configs/pia-cifar10-graybox-assets.local.yaml `
  --workspace workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-256 `
  --repo-root external/PIA `
  --member-split-root external/PIA/DDPM `
  --device cuda:0 `
  --max-samples 256 `
  --batch-size 8 `
  --stochastic-dropout-defense `
  --provenance-status source-retained-unverified
```

## B. Baseline Result

- `workspace = workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-256`
- `auc = 0.841293`
- `asr = 0.78125`
- `tpr@1%fpr = 0.039062`
- `tpr@0.1%fpr = 0.019531`
- `num_samples = 256`
- `elapsed_seconds = 77.248308`

## C. Defense Result

- `workspace = workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-256`
- `defense = stochastic-dropout`
- `auc = 0.82901`
- `asr = 0.767578`
- `tpr@1%fpr = 0.027344`
- `tpr@0.1%fpr = 0.015625`
- `num_samples = 256`
- `elapsed_seconds = 92.490869`

## D. Interpretation

- This pair repeats the same directional effect already seen at `GPU128`.
- The defense run again lowers `AUC`, `ASR`, and both low-FPR TPR metrics relative to the paired baseline.
- The gray-box line now has two consecutive same-path favorable defense signals at `128` and `256` samples.
- That is still not enough to call `G-1` fully validated, but it is enough to treat it as the current formal gray-box defense candidate.

## E. Next Step

1. Promote the gray-box defense language from `prototype` to `provisional G-1`.
2. Write the `GPU256` pair into the status pages and intake manifest.
3. Decide whether the next run should be `GPU512` or a repeat at `256`.
