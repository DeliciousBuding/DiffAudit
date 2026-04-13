# 2026-04-09 Gray-Box Follow-Up: PIA GPU512 Adaptive Ablation

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-09 +08:00`
- `selected_mainline`: `PIA`
- `current_state`: `GPU512 baseline + all_steps + late_steps_only adaptive ablation completed`
- `gpu_usage`: `single GPU, serial`
- `evidence_level`: `runtime-mainline`

## A. Commands Run

Baseline with adaptive review:

```powershell
conda run -n diffaudit-research python -m diffaudit run-pia-runtime-mainline `
  --config tmp/configs/pia-cifar10-graybox-assets.local.yaml `
  --workspace workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260409-gpu-512-adaptive `
  --repo-root external/PIA `
  --member-split-root external/PIA/DDPM `
  --device cuda:0 `
  --max-samples 512 `
  --batch-size 8 `
  --adaptive-query-repeats 3 `
  --provenance-status workspace-verified
```

`all_steps` defense:

```powershell
conda run -n diffaudit-research python -m diffaudit run-pia-runtime-mainline `
  --config tmp/configs/pia-cifar10-graybox-assets.local.yaml `
  --workspace workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260409-gpu-512-allsteps-adaptive `
  --repo-root external/PIA `
  --member-split-root external/PIA/DDPM `
  --device cuda:0 `
  --max-samples 512 `
  --batch-size 8 `
  --stochastic-dropout-defense `
  --dropout-activation-schedule all_steps `
  --adaptive-query-repeats 3 `
  --provenance-status workspace-verified
```

`late_steps_only` defense:

```powershell
conda run -n diffaudit-research python -m diffaudit run-pia-runtime-mainline `
  --config tmp/configs/pia-cifar10-graybox-assets.local.yaml `
  --workspace workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260409-gpu-512-latesteps-adaptive `
  --repo-root external/PIA `
  --member-split-root external/PIA/DDPM `
  --device cuda:0 `
  --max-samples 512 `
  --batch-size 8 `
  --stochastic-dropout-defense `
  --dropout-activation-schedule late_steps_only `
  --adaptive-query-repeats 3 `
  --provenance-status workspace-verified
```

## B. Result Snapshot

| rung | single-query AUC | adaptive AUC | single-query ASR | adaptive ASR | LPIPS surrogate | FID surrogate | wall-clock(s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline `off` | `0.841339` | `0.841339` | `0.786133` | `0.786133` | `0.188377` vs input | `0.014127` vs input | `212.993824` |
| defense `all_steps` | `0.827499` | `0.828075` | `0.770508` | `0.767578` | `0.035881` vs baseline surrogate | `0.000266` vs baseline surrogate | `223.128433` |
| defense `late_steps_only` | `0.836761` | `0.836945` | `0.77832` | `0.77832` | `0.0` vs baseline surrogate | `0.0` vs baseline surrogate | `214.868892` |

Adaptive repeated-query setting:

- `query_repeats = 3`
- `aggregation = mean`

Adaptive score std summary:

- `all_steps`
  - member mean std: `0.230049`
  - non-member mean std: `0.574338`
- `late_steps_only`
  - member mean std: `0.047646`
  - non-member mean std: `0.236998`

## C. Interpretation

- `all_steps` remains the stronger defended setting in privacy terms.
- Its adaptive AUC still stays below baseline by about `0.013264`, so the repeated-query review does **not** wash the effect away.
- `late_steps_only` preserves surrogate quality almost perfectly, but the adaptive AUC only drops by about `0.004394`, which is materially weaker than `all_steps`.
- Current best reading is therefore:
  - `all_steps` = defended mainline candidate under the current `G-1`
  - `late_steps_only` = quality-preserving ablation, not yet strong enough to replace the mainline

## D. Allowed Claim

After this run set, the strongest repository-safe claim is:

- `stochastic-dropout` remains `provisional G-1`
- the new adaptive repeated-query review did **not** fully remove the privacy drop
- `all_steps` is still the strongest defended setting in the current local PIA path
- `late_steps_only` shows that quality can be preserved much better, but its privacy drop is weaker and currently insufficient to replace the mainline

This is still **not** a validated privacy win or a paper-aligned defense benchmark.

## E. Immediate Next Step

1. Promote the canonical gray-box baseline to the new adaptive-reviewed `workspace-verified` summary.
2. Promote `all_steps` as the current defended summary path in the PIA manifest.
3. Write `adaptive_check`, `quality`, `cost`, `provenance_status`, and `defense_stage` into the unified attack-defense table.
4. Keep `late_steps_only` in the gray-box note as the main ablation result.
