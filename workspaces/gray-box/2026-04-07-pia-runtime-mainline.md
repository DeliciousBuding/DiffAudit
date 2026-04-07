# 2026-04-07 Gray-Box Follow-Up: PIA Runtime Mainline On Real Assets

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-07 23:59:00 +08:00`
- `selected_mainline`: `PIA`
- `current_state`: `runtime mainline ready on canonical local assets`
- `gpu_usage`: `not requested`
- `evidence_level`: `runtime-mainline`

## A. Commands Run

Baseline:

```powershell
conda run -n diffaudit-research python -m diffaudit run-pia-runtime-mainline `
  --config tmp/configs/pia-cifar10-graybox-assets.local.yaml `
  --workspace workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260407-cpu `
  --repo-root external/PIA `
  --member-split-root external/PIA/DDPM `
  --device cpu
```

Defense prototype:

```powershell
conda run -n diffaudit-research python -m diffaudit run-pia-runtime-mainline `
  --config tmp/configs/pia-cifar10-graybox-assets.local.yaml `
  --workspace workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260407-cpu `
  --repo-root external/PIA `
  --member-split-root external/PIA/DDPM `
  --device cpu `
  --stochastic-dropout-defense
```

## B. Baseline Result

Canonical summary:

- `workspace`: `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260407-cpu`
- `contract_stage`: `target`
- `asset_grade`: `single-machine-real-asset`
- `provenance_status`: `source-retained-unverified`

Key metrics:

- `auc = 0.90625`
- `asr = 0.875`
- `tpr@1%fpr = 0.75`
- `tpr@0.1%fpr = 0.75`

Runtime facts:

- `num_samples = 8`
- `attack_num = 30`
- `interval = 10`
- `device = cpu`
- `elapsed_seconds = 25.856309`

## C. Defense Prototype Result

Prototype:

- `name = stochastic-dropout`
- `enabled = true`

Key metrics:

- `auc = 0.90625`
- `asr = 0.875`
- `tpr@1%fpr = 0.75`
- `tpr@0.1%fpr = 0.75`

Observed effect in this tiny local run:

- the score threshold shifted
- the ranking metrics did not improve over baseline
- this is therefore only a first runnable defense hook, not a validated privacy win

## D. Interpretation

What is now true:

- `PIA` is no longer only `runtime-preview`
- the repository has a real `runtime-mainline` command that consumes canonical local assets
- the output carries `contract_stage / asset_grade / provenance_status / evidence_level`

What is still not true:

- `paper-aligned`
- `benchmark-ready`
- `team-wide asset closure`

The correct current wording is:

- `single-machine real-asset runtime mainline ready`
