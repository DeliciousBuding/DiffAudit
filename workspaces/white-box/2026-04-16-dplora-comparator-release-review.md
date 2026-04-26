# 2026-04-16 DP-LoRA Comparator Release Review

## Question

After `WB-5` closed the `DP-LoRA` dossier, what is the next honest live question for this successor lane: immediate GPU validation, or a comparator-first release review?

## Inputs

- `workspaces/intake/2026-04-16-phase-e-registry-refresh-and-dplora-selection-verdict.md`
- `workspaces/white-box/2026-04-16-dplora-protocol-overlap-note.md`
- `workspaces/white-box/2026-04-16-dplora-minimal-local-config-candidate.md`
- `workspaces/white-box/2026-04-16-dplora-no-go-and-gpu-release-triggers.md`
- `workspaces/intake/2026-04-14-baseline-smp-lora-w1-comparator-admission-packet.md`
- `outputs/smp-lora-sweep/sweep_results.json`
- `outputs/smp-lora-phase2/baseline_nodefense_target-64/evaluation.json`

## Current Read

What already exists:

- local historical baseline:
  - `AUC = 0.5565217391304348`
- best observed local `SMP-LoRA` point:
  - `lambda=0.1 / rank=4 / epochs=10 / AUC = 0.34375`
- scripts for:
  - local LoRA training
  - local GSA-side evaluation
- an explicit comparator admission packet:
  - `baseline vs SMP-LoRA vs W-1`

What still does **not** exist:

- one fresh comparator packet executed under a locked board
- one same-protocol result against `W-1 strong-v3`
- one honest GPU hypothesis beyond “recheck the historical best point”

## Release Review

The next honest lane is:

- `comparator-first review`
- not immediate GPU release

Why:

1. the historical `SMP-LoRA` local point is strong enough to justify keeping the successor lane alive;
2. but it is still only local bridge evidence, not a new comparator verdict against `W-1`;
3. the repo already has the correct next bounded question written down:
   - `baseline vs SMP-LoRA vs W-1`
4. therefore the highest-value next step is to harden that comparator packet into an execution-ready contract, not to reopen standalone training sweeps.

## Verdict

- `comparator_release_review = positive`
- `next_live_whitebox_question = baseline-vs-smp-lora-vs-w1 comparator`
- `immediate_gpu_release = none`
- `current_verdict = positive but bounded`
- `next_step = keep WB-next on comparator contract hardening rather than direct GPU validation`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this changes white-box queue discipline and next-question truth, but does not change admitted white-box claims.
