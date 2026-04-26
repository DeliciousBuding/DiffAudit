# 2026-04-16 DP-LoRA W-1 Local-Surface Refresh Feasibility

## Question

After `WB-9` required one bounded `W-1` local-surface refresh, can that next step already be executed as an evaluation-only refresh on frozen `strong-v3` artifacts, or would it require retraining or a new bridge build first?

## Inputs

- `workspaces/white-box/2026-04-16-dplora-comparator-schema-alignment-contract.md`
- `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv3-3shadow-full-rerun8-20260408/summary.json`
- `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-sameproto3shadow-batch32-diagnostic-20260409/summary.json`
- `outputs/smp-lora-phase2/baseline_nodefense_target-64/eval_baseline.py`
- `outputs/smp-lora-sweep/lambda0.1_rank4_ep10/config.json`
- `src/diffaudit/cli.py`
- `src/diffaudit/defenses/dpdm_w1.py`

## Provenance Check

### A. Baseline local surface

`baseline_nodefense_target-64/eval_baseline.py` evaluates:

- `workspaces/white-box/assets/gsa-cifar10-1k-3shadow/datasets/target-member`
- `workspaces/white-box/assets/gsa-cifar10-1k-3shadow/datasets/target-nonmember`

and writes the local `63 / 63` result.

### B. Frozen SMP-LoRA local surface

The frozen local candidate config at `lambda0.1_rank4_ep10/config.json` points to the same two dataset directories:

- `.../gsa-cifar10-1k-3shadow/datasets/target-member`
- `.../gsa-cifar10-1k-3shadow/datasets/target-nonmember`

So the current local `baseline / SMP-LoRA` board is already anchored on one reproducible asset surface.

### C. W-1 strong-v3 full-scale surface

The frozen `strong-v3 full-scale` comparator summary already exposes:

- exact target checkpoint
- exact shadow checkpoints
- exact target/shadow dataset directories
- `max_samples = 1000`
- `target_eval_size = 2000`

Those dataset directories are the same legacy:

- `workspaces/white-box/assets/gsa-cifar10-1k-3shadow/datasets/...`

So the remaining mismatch is evaluation scale, not asset-family drift.

## Execution-Surface Check

The current `run-dpdm-w1-multi-shadow-comparator` entrypoint already:

- accepts frozen target/shadow checkpoint paths
- accepts explicit target/shadow dataset directories
- accepts `--max-samples`
- resolves `device` to `cuda` or `cpu`
- writes:
  - `summary.json`
  - `scores.json`
  - `attack-output.txt`
- records:
  - `target_eval_size`
  - `shadow_train_size`
  - checkpoint provenance
  - `max_samples`

`dpdm_w1.py` also shows that `max_samples` truncates each dataset directly before scoring.

This means the next alignment step is already supported as:

- `evaluation-only refresh`

not:

- retraining
- checkpoint regeneration
- same-protocol bridge rebuild

## Strongest Existing Operational Precedent

The batch32 bridge diagnostic already proved the same entrypoint can score a smaller surface:

- `max_samples = 128`
- `target_eval_size = 256`

So there is already repo-local evidence that the defended comparator can be re-rendered on a bounded surface without new training.

## Feasible Next Refresh

The clean bounded refresh is:

1. reuse the frozen `strong-v3 full-scale` target/shadow checkpoints
2. reuse the same legacy `gsa-cifar10-1k-3shadow` target/shadow dataset dirs
3. set:
   - `max_samples = 63`

Expected immediate effect:

- `target_eval_size = 126`
- same asset family as the local `baseline / SMP-LoRA` rows
- one fresh `W-1` local-board summary with auditable provenance

This still does **not** mean:

- admitted `W-1 strong-v3 full-scale` is replaced
- the same-protocol bridge is closed
- `DP-LoRA` becomes admitted

It only means the next comparator-alignment step is operationally ready.

## Verdict

- `local_surface_refresh_feasibility = positive`
- `requires_retraining = no`
- `requires_new_checkpoints = no`
- `requires_new_asset_family = no`
- `recommended_refresh = reuse strong-v3 checkpoints on gsa-cifar10-1k-3shadow dirs with max_samples=63`
- `execution_type = evaluation-only`
- `gpu_release = none`
- `next_step = write one bounded W-1 local-surface refresh execution packet`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this changes only the next white-box execution contract.
