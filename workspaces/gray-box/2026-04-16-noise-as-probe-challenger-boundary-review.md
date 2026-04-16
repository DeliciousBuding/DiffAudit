# 2026-04-16 Noise-as-a-Probe Challenger Boundary Review

## Question

After `GB-14`, what is the honest packaging boundary for the current local `Noise as a Probe` line: should it enter the same gray-box narrative layer as `PIA` and `TMIA-DM`, or should it stay a bounded challenger candidate below the current packaged hierarchy?

## Inputs Reviewed

- `workspaces/gray-box/2026-04-16-noise-as-probe-threshold-hardening-verdict.md`
- `workspaces/gray-box/2026-04-16-noise-as-probe-larger-rung-verdict.md`
- `workspaces/gray-box/2026-04-16-noise-as-probe-larger-rung-repeat-verdict.md`
- `workspaces/gray-box/2026-04-16-pia-vs-tmiadm-operating-point-comparison.md`
- `workspaces/gray-box/2026-04-16-pia-vs-tmiadm-defended-operating-point-comparison.md`
- `workspaces/gray-box/plan.md`

## Current Read

What `Noise as a Probe` now has:

- a real local execution path on `SD1.5 + celeba_partial_target/checkpoint-25000`
- bounded `8 / 8 / 8` positive rung plus disjoint repeat
- bounded `16 / 16 / 16` positive rung plus same-scale disjoint repeat
- a conservative frozen threshold story that remains clean across the stronger bounded rungs

What it still does **not** have:

- same-protocol comparability with admitted `PIA` or current `TMIA-DM` challenger runs
- a defended branch
- a larger benchmark surface
- paper-faithful parity
- a project-level operating-point comparison against current packaged gray-box lines

## Packaging Decision

The honest current packaging boundary is:

- `PIA` remains admitted gray-box headline
- `TMIA-DM` remains the strongest packaged active challenger
- `Noise as a Probe` is now a `strengthened bounded challenger candidate`

Why it does **not** yet replace `TMIA-DM` as the active challenger:

1. `TMIA-DM` already has repeated GPU128/GPU256 evidence and operating-point comparison against `PIA`
2. `TMIA-DM` already has defended-challenger evidence
3. `Noise as a Probe` currently lives on a different local contract surface:
   - `latent-diffusion / SD1.5 / CelebA target-family LoRA`
   - not the admitted `DDPM/CIFAR10` comparison surface
4. current `Noise as a Probe` evidence is strong for bounded local truth, but still too narrow for direct hierarchy replacement

Why it is no longer just a speculative side note:

1. it now has two successful `16 / 16 / 16` rungs
2. the stronger result survived a same-scale disjoint repeat
3. frozen threshold transfer stayed clean rather than collapsing

## Verdict

- `boundary_review_verdict = positive`
- `headline_change = none`
- `active_challenger_change = none`
- `noise_as_probe_status = strengthened bounded challenger candidate`
- `next_step = keep future work on comparison/promotion review or defended extension, not blind rung inflation`

## Handoff Suggestion

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = suggestion-only`

Suggestion:

- current materials and higher-layer summaries should still say:
  - `PIA` = gray-box headline
  - `TMIA-DM` = strongest packaged challenger
- if `Noise as a Probe` is mentioned at all, it should be described as:
  - `new latent-diffusion challenger candidate with repeat-positive bounded local evidence`
  - not as a replacement headline or replacement challenger yet
