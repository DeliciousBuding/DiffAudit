# 2026-04-16 PIA 2048 CDI Rung Runtime-Health Review

## Question

After launching the active `PIA 2048 shared-score surface` GPU rung for the `CDI` lane, is the run still healthy enough to keep alive, or has it already crossed into wasteful silent-hang territory?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260415-gpu-1024-adaptive/summary.json`
- active process state for:
  - `conda env = diffaudit-research`
  - `python -m diffaudit run-pia-runtime-mainline ... --max-samples 2048`
- current `nvidia-smi` snapshots during execution
- current run root:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260416-gpu-2048-cdi-r1`

## Current Facts

### 1. The GPU task is still genuinely alive

- active process:
  - `python`
  - started at `2026-04-16 13:28:51`
- cumulative CPU time kept increasing during review
- `nvidia-smi` continued to show:
  - `cuda:0` occupied by the same `diffaudit-research` Python process
  - memory around `5.2GB .. 5.5GB`
  - non-zero GPU utilization across repeated polls

This is not consistent with an already-dead process.

### 2. But the run is artifact-silent so far

- the expected run root already exists
- but no first-wave artifacts have appeared yet:
  - no `summary.json`
  - no `scores.json`
  - no `adaptive-scores.json`

So the run is still in a pre-write stage.

### 3. Runtime is now beyond the simplest linear expectation

Reference rung:

- `pia-cifar10-runtime-mainline-20260415-gpu-1024-adaptive`
- recorded `wall_clock_seconds = 494.968358`

Current active rung:

- `max_samples = 2048`
- same `adaptive_query_repeats = 3`
- same `batch_size = 8`

Simple scale-up expectation would be around `~2x` the `1024` rung, not an open-ended silent runtime.

The current rung has already run materially longer than the cleanest naive expectation, so it should not be allowed to continue indefinitely without a hard cap.

## Verdict

- `runtime_health_verdict = positive but bounded`
- keep the active `PIA 2048` GPU rung alive for now
- do **not** open another GPU task
- but this rung is now under a hard runtime-health cap rather than open-ended patience

## Carry-Forward Rule

Keep the run alive only while all of the following remain true:

1. the same Python process is still alive
2. `cuda:0` utilization remains non-trivial
3. memory use does not spiral into instability

Stop and classify the rung as `blocked / wasteful silent run` if either occurs:

1. the process disappears without writing the expected artifacts
2. the rung reaches a clearly excessive wall-clock without artifact emission

Current operational cap:

- allow the current rung to continue until roughly `40` minutes wall-clock from launch
- if it still has not emitted first-wave artifacts by then, stop it and record a blocker verdict instead of letting it burn the GPU indefinitely

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

