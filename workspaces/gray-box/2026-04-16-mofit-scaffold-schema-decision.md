# 2026-04-16 MoFit Scaffold / Schema Decision

## Question

For the first dedicated `MoFit` lane, what exact scaffold shape and minimum artifact schema should be frozen before any honest smoke is allowed?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-mofit-protocol-asset-contract.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-mofit-implementation-surface-review.md`
- `<DIFFAUDIT_ROOT>/Research/scripts/run_noise_as_probe_interface_canary.py`
- `<DIFFAUDIT_ROOT>/Research/scripts/run_structural_memorization_smoke.py`
- `<DIFFAUDIT_ROOT>/Research/scripts/run_blackbox_semantic_aux_probe.py`

## Scaffold Decision

Selected scaffold:

- `dedicated script: scripts/run_mofit_interface_canary.py`

Why:

1. `MoFit` owns two method-specific optimization loops:
   - surrogate optimization under null condition
   - fitted-embedding optimization under the target latent path
2. Neither `Noise as a Probe` nor `structural memorization` owns that surface cleanly.
3. A dedicated script keeps current latent-diffusion helpers reusable without polluting existing method semantics.

## Minimum CLI Shape

The first scaffold should freeze these arguments:

- `--run-root`
- `--member-dir`
- `--nonmember-dir`
- `--model-dir`
- `--lora-dir`
- `--blip-dir`
- `--member-limit`
- `--nonmember-limit`
- `--member-offset`
- `--nonmember-offset`
- `--device`

Method-specific arguments that must also be frozen:

- `--surrogate-steps`
- `--embedding-steps`
- `--surrogate-lr`
- `--embedding-lr`
- `--guidance-scale`
- `--max-timestep`
- `--record-trace`

## Minimum Artifact Schema

The first honest MoFit scaffold should emit:

### 1. `summary.json`

Must include at least:

- `task`
- `target_family`
- `caption_source`
- `surrogate_steps`
- `embedding_steps`
- `member_count`
- `nonmember_count`
- `artifact_schema_version`

### 2. `records.jsonl`

One row per sample, with at least:

- `split`
- `file_name`
- `prompt_source`
- `prompt_text`
- `surrogate_trace_path`
- `embedding_trace_path`
- `l_cond`
- `l_uncond`
- `mofit_score`

### 3. trace artifacts

At minimum:

- one surrogate optimization trace per sample
- one fitted-embedding optimization trace per sample

The trace format can stay lightweight for now:

- JSON list of step-level scalar summaries is enough

## Carry-Forward Rule

- the scaffold should be implemented before any smoke
- the first implementation task should stay:
  - `CPU-only`
  - `non-run`
  - `gpu_release = none`
- do not let the first scaffold silently skip:
  - surrogate traces
  - fitted-embedding traces
  - `L_cond / L_uncond / mofit_score`

## Verdict

- `scaffold_schema_verdict = positive`
- `MoFit` now has a frozen dedicated scaffold choice
- `MoFit` now has a frozen minimum artifact schema
- `next_step = implement dedicated scaffold script with the frozen schema`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this is method-entry and artifact-shape truth only.
