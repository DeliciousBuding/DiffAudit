# 2026-04-16 White-Box Feature Trajectory Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `WB-4 / white-box feature-trajectory upgrade`
- `device`: `cpu`
- `decision`: `negative but useful`

## Hypothesis

If the migrated `Finding NeMo` DDPM observability route no longer resolves cleanly on current admitted white-box assets, then the white-box feature/trajectory branch would need renewed setup work; if it still resolves cleanly, the honest question becomes whether that readiness changes white-box story at all.

## Executed Probe

CPU-only read-only contract probe:

```powershell
conda run -n diffaudit-research python -m diffaudit probe-gsa-observability-contract `
  --repo-root <DIFFAUDIT_ROOT>/Research/workspaces/white-box/external/GSA `
  --assets-root <DIFFAUDIT_ROOT>/Research/workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1 `
  --checkpoint-root <DIFFAUDIT_ROOT>/Research/workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/checkpoints/target `
  --split target-member `
  --sample-id target-member/00-data_batch_1-00965.png `
  --layer-selector mid_block.attentions.0.to_v `
  --signal-type activations `
  --resolution 32
```

## Result

Observed probe payload:

- `status = ready`
- all current contract checks resolved:
  - workspace files
  - assets root
  - checkpoint root
  - sample binding
  - layer selector
- resolved checkpoint:
  - `checkpoint-9600`
- resolved selector:
  - `mid_block.attentions.0.to_v`

## Verdict

Current verdict:

- `negative but useful`

Reason:

1. the current feature/trajectory observability path is still available as a bounded CPU-only contract probe;
2. that means the branch is not blocked by missing entry plumbing;
3. but it still does **not** change white-box story:
  - no GPU release
  - no validation-smoke release
  - no admitted change
  - no benchmark or mechanism claim
4. so the only honest update is “the observability route remains ready below release threshold,” not “white-box story has expanded.”

## Decision

Current decision:

- close `WB-4` for the current wave as `no story change`
- keep the observability route on `zero-GPU hold`
- do not spend more white-box budget on feature/trajectory work unless a separate release review changes the threshold

## Handoff Note

- `Platform`: no sync needed.
- `Runtime`: no sync needed.
- `Materials`: if mentioned, the honest wording is that the observability route remains available but still sits below run-release and benchmark-release boundaries.
