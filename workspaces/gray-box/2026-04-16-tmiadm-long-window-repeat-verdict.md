# 2026-04-16 TMIA-DM Long-Window Repeat Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `GB-3 / TMIA-DM long_window repeat`
- `selected_family`: `TMIA-DM long_window`
- `gpu_status`: `not requested`
- `decision`: `repeat-positive but not gpu-ready`

## Question

After the first `TMIA-DM` protocol probe showed a positive `long_window` branch, does that branch stay positive under a bounded same-asset repeat with a different noise seed?

## Executed Evidence

Primary probe:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-protocol-probe-20260416-cpu-32-r1/summary.json`

Repeat probe:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-protocol-probe-20260416-cpu-32-r2-seed1/summary.json`

Prior protocol verdict:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-tmiadm-protocol-probe-verdict.md`

## Metrics

`TMIA-DM long_window` on matched `CPU-32` probes:

- `r1 / seed0`:
  - `AUC = 0.702148`
  - `ASR = 0.703125`
  - `TPR@1%FPR = 0.03125`
- `r2 / seed1`:
  - `AUC = 0.663086`
  - `ASR = 0.671875`
  - `TPR@1%FPR = 0.0625`

Family ordering across both repeats:

- `long_window` stayed best on both runs
- `short_window` stayed negative on both runs
- naive `fused` stayed near-random on both runs

## Verdict

Current verdict:

- `repeat-positive but not gpu-ready`

Reason:

1. `long_window` stayed directionally positive across two bounded runs on the same admitted asset line;
2. its quality moved down from `AUC = 0.702148` to `AUC = 0.663086`, so stability is real but only moderate;
3. the rest of the family did not improve: `short_window` remained negative and naive fusion remained non-useful;
4. even the better rung still trails the current `PIA cpu-32` baseline, so this is not yet a challenger-grade release decision.

Interpretation:

- the positive `TMIA-DM` signal is not a single-seed fluke;
- but the current evidence still supports only a narrow `long_window` branch, not the family as a whole;
- the repo should therefore treat `TMIA-DM long_window` as an active bounded refinement line, still below GPU release threshold.

## Decision

Current release decision:

- `no GPU release yet`
- `keep long_window as active bounded branch`
- `do not reopen short_window`
- `do not promote naive fusion`

Meaning:

1. the next bounded step should refine or stress-test `long_window`, not broaden prematurely;
2. a GPU request still needs either stronger repeat stability or a cleaner same-split comparison against `PIA`;
3. the current family is promising enough to keep alive, but not strong enough to headline.

## Next Gate

The next bounded `TMIA-DM` gate should be one of:

1. a second same-budget repeat with another seed or tighter timestep subset;
2. a same-split comparison note between `PIA` and `TMIA-DM long_window`;
3. a bounded feature-refinement attempt that changes only `long_window`, not the whole family.

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: no direct change needed.
- Materials: wording can now say `TMIA-DM long_window` survived one bounded repeat, but should still be described as a refinement branch rather than a GPU-released challenger.
