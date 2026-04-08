# 2026-04-08 Gray-Box Follow-Up: PIA GPU128 Attack-Defense Pair

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-08 21:40:30 +08:00`
- `selected_mainline`: `PIA`
- `current_state`: `single-GPU baseline and stochastic-dropout defense pair completed at 128 samples`
- `gpu_usage`: `single GPU, serial`
- `evidence_level`: `runtime-mainline`

## A. Commands Run

Baseline:

```powershell
conda run -n diffaudit-research python -m diffaudit run-pia-runtime-mainline `
  --config tmp/configs/pia-cifar10-graybox-assets.local.yaml `
  --workspace workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-128 `
  --repo-root external/PIA `
  --member-split-root external/PIA/DDPM `
  --device cuda:0 `
  --max-samples 128 `
  --batch-size 8 `
  --provenance-status source-retained-unverified
```

Defense:

```powershell
conda run -n diffaudit-research python -m diffaudit run-pia-runtime-mainline `
  --config tmp/configs/pia-cifar10-graybox-assets.local.yaml `
  --workspace workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-128 `
  --repo-root external/PIA `
  --member-split-root external/PIA/DDPM `
  --device cuda:0 `
  --max-samples 128 `
  --batch-size 8 `
  --stochastic-dropout-defense `
  --provenance-status source-retained-unverified
```

## B. Baseline Result

- `workspace = workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-128`
- `auc = 0.817444`
- `asr = 0.765625`
- `tpr@1%fpr = 0.046875`
- `tpr@0.1%fpr = 0.039062`
- `num_samples = 128`
- `elapsed_seconds = 39.446178`

## C. Defense Result

- `workspace = workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-128`
- `defense = stochastic-dropout`
- `auc = 0.803955`
- `asr = 0.757812`
- `tpr@1%fpr = 0.03125`
- `tpr@0.1%fpr = 0.015625`
- `num_samples = 128`
- `elapsed_seconds = 44.112771`

## D. Interpretation

- This is the first same-path `GPU128` gray-box comparison under the current canonical PIA entrypoint.
- The defense run lowers `AUC`, `ASR`, and both low-FPR TPR metrics relative to the paired baseline.
- That is enough to call this a first favorable defense signal.
- It is not enough to call `G-1` validated yet, because this is still one local pair on a `source-retained-unverified` asset line.

## E. Next Step

1. Write this pair into the gray-box status pages and intake manifest.
2. Repeat or scale the same pair to `256` samples.
3. Decide whether `stochastic-dropout` can now be formalized as provisional `G-1`.
