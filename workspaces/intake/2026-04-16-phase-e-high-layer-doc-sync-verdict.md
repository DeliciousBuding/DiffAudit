# 2026-04-16 Phase E High-Layer Doc Sync Verdict

## Question

After `phase-e-candidates.json` was refreshed into a sparse-hold registry, which high-layer `Phase E` documents were still carrying stale candidate-ordering language, and what exact wording needed to change?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/intake/phase-e-candidates.json`
- `<DIFFAUDIT_ROOT>/Research/docs/future-phase-e-intake.md`
- `<DIFFAUDIT_ROOT>/Research/docs/reproduction-status.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/intake/2026-04-16-phase-e-sparse-registry-refresh-verdict.md`

## Stale Points

1. `docs/future-phase-e-intake.md` still presented:
   - `DP-LoRA`
   - `SecMI unblock`
   - `TMIA-DM intake`
   as if they remained on the current intake-only ordering surface.

2. `docs/reproduction-status.md` still described:
   - old `SMP-LoRA T06 optimizer/lr frontier` queue priority
   - `SecMI = blocked baseline`
   - `TMIA-DM = intake`
   even though current repository truth has already moved on.

## Sync Decision

- keep `PIA paper-aligned confirmation` as document-layer conditional only
- keep `Finding NeMo` as the only remaining intake-only candidate, explicitly under `zero-GPU hold`
- remove `DP-LoRA`, `SecMI unblock`, and `TMIA-DM intake` from current `Phase E` candidate-ordering wording
- rewrite `DP-LoRA / SMP-LoRA` as:
  - `bounded exploration branch`
  - `no-new-gpu-question`
- rewrite `SecMI` as:
  - `independent corroboration line`
- rewrite `TMIA-DM` as:
  - `strongest packaged gray-box challenger`

## Verdict

- `sync_verdict = positive`
- the two highest-value high-layer `Phase E` docs are now aligned to the sparse-hold registry
- further broader doc cleanup is optional follow-up, not required for this round

## Handoff / Sync Decision

- `Research/ROADMAP.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
