# 2026-04-17 Finding NeMo Intervention-On/Off Bounded Review Surface Verdict

## Question

After `I-B.9` froze the first honest contract, can the repository actually expose a bounded intervention-on/off review surface on admitted `GSA` assets without overclaiming that the defense result already exists?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-first-intervention-on-off-bounded-review-contract-selection.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-bounded-attack-side-evaluation-packet-control-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/cli.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/gsa.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/gsa_observability.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_gsa_adapter.py`

## Implementation

The repository now exposes a dedicated bounded dual-run review surface:

- command:
  - `run-gsa-runtime-intervention-review`

The command now does all of the following in one place:

1. reads a frozen target-anchored mask object from an existing `inmodel-packet-export` summary
2. runs one local baseline extraction board on admitted `GSA` assets
3. runs one local intervened extraction board on the same packet definition
4. evaluates both boards with the same bounded attack-side surface
5. emits:
   - `baseline.metrics`
   - `intervened.metrics`
   - `metric_deltas`
   - `locality_anchor`

Current bounded scope:

- same admitted `GSA` family
- same `max_samples` bounded board
- same frozen `channel_indices`
- no per-model mask reselection

## Verification

Test command:

```powershell
conda run -n diffaudit-research python -m unittest <DIFFAUDIT_ROOT>/Research/tests/test_gsa_adapter.py
```

Result:

- `4` tests pass
- the new CLI command is covered
- the test verifies summary assembly for:
  - frozen mask input
  - baseline/intervened metrics
  - delta fields

## What Landed

This step lands the **review surface**, not the admitted defense verdict.

What is now true:

1. the repo no longer needs to fake an intervention review by stitching together two unrelated artifacts
2. the first honest joint review shape now has a concrete executable entrypoint
3. the joint summary can carry both:
   - bounded attack-side metric movement
   - the existing local canary/control drift anchor

## Boundary

This is still **not** a defense-positive result.

What it proves:

- `I-B.10` implementation exists
- the first honest bounded intervention-on/off review surface is executable in repository code
- the repo now has one concrete place to run the next real packet

What it does **not** prove:

- admitted intervention-on/off metric movement on real assets
- low-FPR improvement under the fixed locality budget
- quality-preserving defense effect
- any competition-facing claim change

## Verdict

- `finding_nemo_intervention_on_off_bounded_review_surface_verdict = positive but bounded`

More precise reading:

1. `I-B.10` is now satisfied:
   - the bounded intervention-on/off review surface is implemented
2. the honest reading is:
   - `implementation-positive / admitted-execution-pending`
3. the next task is no longer surface design:
   - it is execution-budget review and first actual admitted packet launch.

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-intervention-on-off-bounded-review-surface-verdict.md`

## Next Step

- `next_live_cpu_first_lane = I-B.11 execution-budget review for first admitted target-anchored fixed-mask intervention-on/off bounded packet`
- `next_gpu_candidate = I-B.11 actual target-anchored fixed-mask intervention-on/off bounded packet on admitted GSA assets (pending execution-budget review)`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: update required
- `workspaces/implementation/challenger-queue.md`: update required
- `workspaces/white-box/plan.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
