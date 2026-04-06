# 2026-04-07 Gray-Box Follow-Up: PIA Real-Asset Probe

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-07 17:55:00 +08:00`
- `selected_mainline`: `PIA`
- `current_state`: `template-plan ready + real-asset probe blocked`
- `gpu_usage`: `not requested`
- `evidence_level`: `template-probe`

## A. What Was Run

This follow-up recorded the current behavior of the shared `PIA` template config without pretending that real assets already exist.

Commands:

```powershell
conda run -n diffaudit-research python -m diffaudit plan-pia `
  --config configs/attacks/pia_plan.yaml

conda run -n diffaudit-research python -m diffaudit probe-pia-assets `
  --config configs/attacks/pia_plan.yaml `
  --member-split-root external/PIA/DDPM

conda run -n diffaudit-research python -m diffaudit dry-run-pia `
  --config configs/attacks/pia_plan.yaml `
  --repo-root external/PIA `
  --member-split-root external/PIA/DDPM
```

Recorded artifacts:

- [plan.json](runs/pia-followup-20260407/plan.json)
- [probe.json](runs/pia-followup-20260407/probe.json)
- [dry-run.json](runs/pia-followup-20260407/dry-run.json)

## B. What It Proves

1. `plan-pia` succeeds on the shared template, so the planner and CLI entrypoint are intact.
2. `probe-pia-assets` returns `blocked`, but it confirms that `external/PIA/DDPM/CIFAR10_train_ratio0.5.npz` is already present.
3. `dry-run-pia` also returns `blocked`, while confirming that `external/PIA/DDPM/attack.py`, `components.py`, and `model.py` are present and expose the expected `PIA` / `UNet` markers.

The remaining missing items are still:

- real `checkpoint`
- real `dataset_root`
- real `dataset_root/cifar10` layout

## C. Why GPU Was Not Requested

This phase was intentionally CPU-only.

Reason:

- the current bottleneck is asset readiness, not compute
- both `probe-pia-assets` and `dry-run-pia` are file-system / import-level checks
- re-running a GPU synthetic smoke would only repeat a conclusion the project already has

## D. Shortest Next Step

1. Bind a non-template config that points to a real CIFAR10 DDPM checkpoint directory and a real dataset parent directory.
2. Re-run `probe-pia-assets` with that config.
3. If the probe turns `ready`, run `runtime-probe-pia --device cpu`.
4. Only after that, decide whether any GPU work is justified.
