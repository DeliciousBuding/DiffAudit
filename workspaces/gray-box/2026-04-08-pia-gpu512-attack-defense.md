# 2026-04-08 Gray-Box Follow-Up: PIA GPU512 Attack-Defense Pair

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-08 23:17:22 +08:00`
- `selected_mainline`: `PIA`
- `current_state`: `single-GPU baseline and stochastic-dropout defense pair completed at 512 samples`
- `gpu_usage`: `single GPU, serial`
- `evidence_level`: `runtime-mainline`

## A. Commands Run

Baseline:

```powershell
conda run -n diffaudit-research python -m diffaudit run-pia-runtime-mainline `
  --config tmp/configs/pia-cifar10-graybox-assets.local.yaml `
  --workspace workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-512 `
  --repo-root external/PIA `
  --member-split-root external/PIA/DDPM `
  --device cuda:0 `
  --max-samples 512 `
  --batch-size 8 `
  --provenance-status source-retained-unverified
```

Defense:

```powershell
conda run -n diffaudit-research python -m diffaudit run-pia-runtime-mainline `
  --config tmp/configs/pia-cifar10-graybox-assets.local.yaml `
  --workspace workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-512 `
  --repo-root external/PIA `
  --member-split-root external/PIA/DDPM `
  --device cuda:0 `
  --max-samples 512 `
  --batch-size 8 `
  --stochastic-dropout-defense `
  --provenance-status source-retained-unverified
```

## B. Baseline Result

- `workspace = workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-512`
- `auc = 0.841339`
- `asr = 0.786133`
- `tpr@1%fpr = 0.058594`
- `tpr@0.1%fpr = 0.011719`
- `num_samples = 512`
- `elapsed_seconds = 171.214752`

## C. Defense Result

- `workspace = workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-512`
- `defense = stochastic-dropout`
- `auc = 0.82938`
- `asr = 0.769531`
- `tpr@1%fpr = 0.023438`
- `tpr@0.1%fpr = 0.009766`
- `num_samples = 512`
- `elapsed_seconds = 131.89636`

## D. Interpretation

- This pair extends the same-path comparison to `512` samples.
- The defense run again lowers `AUC`, `ASR`, and both low-FPR TPR metrics relative to the paired baseline.
- The gray-box line now has three consecutive favorable defense signals at `128`, `256`, and `512`.
- That is enough to formalize `stochastic-dropout` as the current `provisional G-1`.
- It is still not enough to call `G-1` validated, because the asset line remains `source-retained-unverified` and there is still no repeat run at the same scale.

## E. Next Step

1. Promote the gray-box defense language from `prototype` to `provisional G-1`.
2. Write the `GPU512` pair into the status pages and intake manifest.
3. Decide whether the next gray-box action is a repeat at `512` or a pivot to `SecMI` promote/block.
