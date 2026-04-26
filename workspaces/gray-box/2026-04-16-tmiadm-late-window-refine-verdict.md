# 2026-04-16 TMIA-DM Late-Window Refine Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `GB-3 / TMIA-DM late-window refine`
- `selected_family`: `TMIA-DM long_window`
- `selected_window`: `[80, 100, 120]`
- `gpu_status`: `not yet requested`
- `decision`: `competitive-positive and gpu-eligible`

## Question

If the current `TMIA-DM` signal is restricted to a later timestep window, does `long_window` become strong enough to justify a bounded GPU pilot rather than staying a CPU-only refinement branch?

## Executed Evidence

Late-window bounded probes:

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\tmiadm-cifar10-late-window-protocol-probe-20260416-cpu-32-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\tmiadm-cifar10-late-window-protocol-probe-20260416-cpu-32-r2-seed1\summary.json`

Comparison references:

- `PIA cpu-32 baseline`:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-cifar10-runtime-mainline-20260408-cpu-32\summary.json`
- prior full-window repeat note:
  - `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-tmiadm-long-window-repeat-verdict.md`

## Metrics

Late-window `TMIA-DM long_window`:

- `r1 / seed0`:
  - `AUC = 0.823242`
  - `ASR = 0.796875`
  - `TPR@1%FPR = 0.28125`
- `r2 / seed1`:
  - `AUC = 0.760742`
  - `ASR = 0.75`
  - `TPR@1%FPR = 0.0625`

Current aligned `PIA cpu-32` reference:

- `AUC = 0.782227`
- `ASR = 0.765625`
- `TPR@1%FPR = 0.09375`

Readout:

- one late-window repeat beat `PIA` clearly;
- the second late-window repeat softened, but stayed close enough to remain competitive;
- both late-window runs were materially stronger than the earlier full-window `TMIA-DM` path.

## Verdict

Current verdict:

- `competitive-positive and gpu-eligible`

Reason:

1. restricting `TMIA-DM long_window` to the late window produced a large uplift over the earlier full-window protocol probe;
2. the uplift survived a same-budget repeat, even though the second run softened;
3. this is the first `TMIA-DM` local configuration that plausibly competes with `PIA` instead of merely trailing it;
4. the evidence is still too narrow to replace `PIA` as the gray-box headline, but it is strong enough to justify a bounded GPU pilot.

Interpretation:

- `TMIA-DM` is no longer only a secondary corroboration branch on CPU;
- the late-window `long_window` variant is now the first real gray-box challenger candidate inside this family;
- the correct next step is a minimal GPU rung, not another broad family expansion.

## Decision

Current release decision:

- `allow minimal GPU pilot`
- `keep PIA as current headline until GPU evidence lands`
- `keep short_window/fused closed`

Meaning:

1. a bounded `GPU128` or similarly small first rung is now justified;
2. the pilot should keep the exact late-window contract rather than reopening full-window search;
3. material wording may now call this branch `gpu-eligible`, but not yet `promoted`.

## Next Gate

The next task should be:

1. launch one bounded `TMIA-DM late-window` GPU pilot on the current CIFAR-10 DDPM asset line;
2. compare that GPU rung directly against the existing `PIA` ladder before any headline change;
3. only then decide whether `TMIA-DM` deserves challenger promotion.

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: no direct change needed.
- Materials: wording can now say `TMIA-DM late-window long_window` is the first GPU-eligible challenger candidate in that family, while still keeping `PIA` as the admitted gray-box headline until GPU evidence arrives.
