# 2026-04-17 Finding NeMo First Admitted Fixed-Mask Packet Execution-Budget Review

## Question

After `I-B.10` landed the dual-run review surface, is the first admitted target-anchored fixed-mask intervention-on/off packet honest to release on the current host as a bounded GPU question?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-finding-nemo-first-intervention-on-off-bounded-review-contract-selection.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-finding-nemo-intervention-on-off-bounded-review-surface-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\assets\gsa-cifar10-1k-3shadow-epoch300-rerun1\manifests\cifar10-ddpm-1k-3shadow-epoch300-rerun1.json`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-runtime-mainline-20260409-cifar10-1k-3shadow-epoch300-rerun1\summary.json`
- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\gsa.py`

## Execution-Budget Facts

### 1. The admitted asset family is still large

Current admitted split counts:

- target member: `1000`
- target nonmember: `1000`
- each shadow member split: `1000`
- each shadow nonmember split: `1000`
- shadow pairs: `3`

So one full attack-side board touches:

- `2` target splits
- `6` shadow splits
- total = `8000` images

### 2. The current dual-run surface repeats that board twice

`run-gsa-runtime-intervention-review` currently executes:

1. one baseline board
2. one intervened board

So the current first admitted packet would traverse:

- `8000` images for baseline
- `8000` images for intervened
- total = `16000` image-level gradient extractions

### 3. The current implementation is not extraction-bounded

Current execution path:

- `_iter_dataset_files(...)` enumerates every file in each dataset directory
- `_extract_gsa_gradients_with_fixed_mask(...)` loops over every returned file
- `run_gsa_runtime_intervention_review(...)` invokes that extractor for all target/shadow jobs on both baseline and intervened branches

The current `max_samples` field only applies later, at:

- `_evaluate_gsa_closed_loop(...)`
- via `_preprocess(...)`

Therefore:

- evaluation is bounded
- extraction is **not** bounded

## Why This Matters

The first admitted packet was frozen as a bounded review question.

But on current code, launching it now would quietly mean:

1. a full admitted-board extraction cost;
2. doubled again for baseline/intervened;
3. while only the downstream evaluation table is small.

That is not the most honest reading of a "first bounded packet", and it is not the best use of a scarce single-GPU slot on the current host.

## Comparison Against Existing Evidence

The admitted baseline run already shows this family is not cheap even before dual-run replay:

- it used full admitted `1k / 1k / 3-shadow`
- it used `ddpm_num_steps = 1000`
- it used `sampling_frequency = 10`
- multiple per-split extractions took on the order of minutes even before any dual-run replay

The current dual-run surface defaults are smaller than the full admitted baseline, but that does not fix the core issue:

- the extraction surface is still full-dataset
- the boundedness currently exists only in the closed-loop evaluation stage

## Verdict

- `finding_nemo_first_admitted_fixed_mask_packet_execution_budget_review_verdict = blocked but useful`

More precise reading:

1. the next GPU candidate is **not** honest to launch yet;
2. the blocker is not attack logic:
   - it is execution budgeting and host-fit;
3. the specific blocker is:
   - missing extraction-side bounded dataset cap on the new dual-run review surface.

## Consequence

The current GPU candidate should return to:

- `next_gpu_candidate = none`

because the packet is still missing one bounded-execution gate.

## Canonical Evidence Anchor

- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-finding-nemo-first-admitted-fixed-mask-packet-execution-budget-review.md`

## Next Step

- `next_live_cpu_first_lane = I-B.12 implement extraction-side bounded dataset cap for target-anchored fixed-mask intervention review`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: update required
- `workspaces/implementation/challenger-queue.md`: update required
- `workspaces/white-box/plan.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
