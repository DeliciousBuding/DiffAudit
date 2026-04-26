# 2026-04-16 DP-LoRA No-Go and GPU Release Triggers

## Question

Under what exact conditions should the current `DP-LoRA` successor lane remain blocked, and what must be true before any future GPU request is honest?

## Inputs

- `workspaces/intake/2026-04-12-smp-lora-gpu-expansion-program.md`
- `workspaces/white-box/2026-04-16-dplora-protocol-overlap-note.md`
- `workspaces/white-box/2026-04-16-dplora-minimal-local-config-candidate.md`

## Current Posture

The current lane is:

- `CPU-first dossier only`
- `not admitted`
- `not same-protocol with W-1`
- `not queue released`
- `gpu_release = none`

## No-Go Triggers

The lane stays `no-go for GPU` if any of the following is still true:

1. `canonical comparator missing`
   - the board is not locked to:
     - `baseline`
     - frozen `SMP-LoRA` local candidate
     - `W-1 strong-v3` reference
2. `evaluation protocol still ambiguous`
   - attacker, split contract, metrics, seed, or output schema remain movable
3. `candidate drift`
   - the local candidate is no longer the frozen
     - `lambda=0.1 / rank=4 / epochs=10`
   - and the new proposal is only another rescue sweep
4. `artifact incompleteness`
   - the proposed run cannot guarantee:
     - final checkpoint pointer
     - config record
     - seed record
     - paired evaluation summary
5. `story drift`
   - the proposal starts implying:
     - paper-faithful `DP-LoRA`
     - direct `W-1` replacement
     - admitted white-box upgrade
6. `hypothesis inflation`
   - the ask becomes a wide sweep instead of one bounded question

## Future GPU Release Triggers

A future GPU request becomes honest only if **all** of the following are true:

1. `single bounded hypothesis`
   - exactly one question beyond the frozen local config is being tested
2. `locked comparator board`
   - `baseline vs SMP-LoRA local candidate vs W-1 strong-v3` is explicitly frozen
3. `locked eval contract`
   - attack panel, split, metrics, seed, and output schema are named in advance
4. `bounded artifact expectation`
   - the run is expected to produce one auditable packet:
     - checkpoint pointer
     - summary/config/seed
     - paired baseline/reference comparison
5. `bounded machine budget`
   - one active GPU task only
   - host impact is acceptable
6. `story boundary preserved`
   - the request still says:
     - `candidate validation`
     - not `admitted upgrade`
     - not `same-protocol replacement`

## Practical Reading

This means the current lane is ready for:

- future release-review preparation

But it is still not ready for:

- immediate GPU allocation
- wide hyperparameter rescue
- replacement of `W-1`

## Verdict

- `trigger_verdict = positive`
- `current_verdict = positive but bounded`
- `current_no_go = explicit`
- `future_gpu_release = conditional only`
- `gpu_release = none`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this sharpens internal release discipline only.
