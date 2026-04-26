# 2026-04-16 Phase E Registry Refresh and DP-LoRA Selection Verdict

## Question

After `BB-6` closed and `Research/ROADMAP.md` returned to "no currently open top-priority lane", which remaining candidate should become the next bounded live task?

## Inputs Reviewed

- `Research/ROADMAP.md`
- `workspaces/white-box/plan.md`
- `workspaces/gray-box/plan.md`
- `workspaces/black-box/plan.md`
- `workspaces/intake/phase-e-candidates.json`
- `workspaces/intake/2026-04-10-phase-e-finding-nemo-intake-hold-decision.md`
- `workspaces/intake/2026-04-10-dplora-comparability-intake.md`
- `workspaces/implementation/2026-04-10-rtx4070-8gb-long-horizon-plan.md`

## Current Truth Before Reselection

1. `BB-6` is complete, so black-box no longer has a fresh execution-ready top lane.
2. Gray-box explicitly says the next step should be another real mechanism or another lane, not more `TMIA-DM` mechanical rungs.
3. White-box explicitly says current budget should move to candidate generation or import, because `WB-3` closed with `none selected`.
4. `Finding NeMo` is still `adapter-complete zero-GPU hold`, and its own governance note requires a separate hypothesis/budget review before any reopen.
5. `DP-LoRA` already has an identified CPU-first dossier shape:
   - `protocol overlap note`
   - `minimal config candidate`
   - `no-go triggers`
6. `phase-e-candidates.json` is stale:
   - it still places `TMIA-DM` in the intake-only candidate surface even though that line has already promoted into an executed gray-box challenger branch.

## Selection Review

### Candidate A: `Finding NeMo`

- Pros:
  - still the richest white-box intake dossier
  - adapter and observability code paths exist
- Why not now:
  - the current hold decision is intentionally strict
  - reopening it honestly requires a fresh hypothesis/budget review first
  - that means it is not the shortest next bounded task

### Candidate B: `DP-LoRA`

- Pros:
  - directly addresses the white-box "no second defended family" gap at the candidate-generation layer
  - already has a defined CPU-first dossier shape in the long-horizon plan
  - can produce a concrete yes/no intake verdict without occupying GPU
- Risk:
  - protocol overlap with admitted `DDPM/CIFAR-10 + GSA/W-1` is incomplete
- Why selected:
  - it is the shortest honest path from "white-box breadth frozen" to "next candidate judged with concrete comparability criteria"

### Candidate C: `SecMI unblock`

- Why not now:
  - still primarily asset-blocked
  - weaker blocker leverage than the white-box candidate-generation gap

## Verdict

- `selection_verdict = positive`
- `selected_next_live_lane = WB-5 DP-LoRA comparability dossier`
- `task_shape = CPU-first dossier / intake hardening`
- `gpu_release = none`
- `next_gpu_candidate = none until the dossier writes protocol overlap, minimal local config, and explicit no-go triggers`

## Registry Decision

1. Promote `DP-LoRA` to intake review priority `#1`.
2. Keep `Finding NeMo` in the registry, but demote it behind `DP-LoRA` because it remains on a stricter `zero-GPU hold`.
3. Remove `TMIA-DM` from the `Phase E` candidate ordering because it is no longer an intake-only candidate surface item.

## Handoff / Sync Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this step changes research queue truth, but does not change admitted results, system-facing fields, or competition claims.
