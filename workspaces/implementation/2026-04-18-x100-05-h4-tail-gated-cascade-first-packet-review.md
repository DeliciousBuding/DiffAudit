# 2026-04-18 X-100 05 H4 Tail-Gated Cascade First Packet Review

## Question

After `X-99` confirmed stable `05-H1/H2` tail lift on the enlarged `461 / 474` pairboard, does the first bounded `H4 tail-gated cascade` packet become a promotable next-step line, or only a bounded auxiliary contract?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\crossbox_pairboard.py`
- `D:\Code\DiffAudit\Research\src\diffaudit\cli.py`
- `D:\Code\DiffAudit\Research\tests\test_crossbox_pairboard.py`
- `D:\Code\DiffAudit\Research\docs\report-bundles\gpt54\round2-results\05.md`
- `D:\Code\DiffAudit\Research\workspaces\cross-box\runs\crossbox-pairboard-gsa-targeted-full-overlap-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\cross-box\runs\crossbox-pairboard-gsa-targeted-full-overlap-h4-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\cross-box\runs\crossbox-pairboard-gsa-targeted-full-overlap-h4-weighted-20260418-r1\summary.json`

## What Landed

### 1. Pairboard now supports bounded `H4 tail-gated cascade`

The repo now exposes one minimal `H4` evaluator on top of cached pairboard surfaces:

- fold-local isotonic probability calibration
- anchor threshold frozen at calibration-side `1%FPR`
- route band selected by routed-fraction quantiles
- conflict discount `gamma` grid

This stays CPU-only and reuses cached score surfaces exactly as required by the report.

### 2. First real `H4` packet was executed on the enlarged full-overlap board

Two bounded variants were run on the same `461 member / 474 nonmember` shared board:

1. `anchor = gsa`, routed candidate = `logistic_2feature`
2. `anchor = gsa`, routed candidate = `weighted_average`

## Actual Read

### Primary `H4` packet: routed `logistic_2feature`

Selected config:

- target route fraction = `30%`
- actual test routed fraction = `0.320513`
- `gamma = 0.1`
- relative overhead = `0.080128`

Primary held-out board:

- `tail_gated_cascade AUC = 0.693782`
- `tail_gated_cascade ASR = 0.636752`
- `tail_gated_cascade TPR@1%FPR = 0.129870`
- `tail_gated_cascade TPR@0.1%FPR = 0.064935`

Repeated-holdout mean deltas versus `best_single`:

- `AUC = -0.105345`
- `ASR = -0.102564`
- `TPR@1%FPR = +0.016450`
- `TPR@0.1%FPR = +0.046753`

### Supporting `H4` packet: routed `weighted_average`

Repeated-holdout mean deltas versus `best_single`:

- `AUC = -0.094104`
- `ASR = -0.085897`
- `TPR@1%FPR = +0.012121`
- `TPR@0.1%FPR = +0.054545`

### Honest reading

This is not a promoted next-stage line.

What it does show:

1. `H4` can recover some low-FPR tail while keeping relative overhead around `8%`
2. so the cost-saver interpretation is real

What it does **not** show:

1. it does not beat the already-promoted always-on `logistic_2feature` fusion
2. it gives up too much `AUC / ASR` to count as a clean promotion
3. the current board no longer supports the old assumption that `GSA` is the strongest standalone anchor, because `best_single = pia` in `5/5` repeats

So the report’s own bounded-auxiliary clause is exactly where this lands:

- `H4 can be a cost-saver`
- `H4 is not a performance booster`

## Verdict

- `x100_05_h4_tail_gated_cascade_verdict = negative but useful`

More precise reading:

1. `05-H1/H2` remains the promoted cross-box result
2. the first bounded `H4` packet is real, but only as auxiliary cost-sensitive routing evidence
3. `H4` should not become the next promoted `05` subline on current assets
4. the immediate active slot can now yield from `05` to `04-defense`

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `current promoted 05 candidate = logistic_2feature`
- `current 05 H4 state = auxiliary-only cost-saver packet; no promotion`
- `next near-term active slot = 04-defense`

## Canonical Evidence Anchor

Primary anchor:

- `D:\Code\DiffAudit\Research\workspaces\cross-box\runs\crossbox-pairboard-gsa-targeted-full-overlap-h4-20260418-r1\summary.json`

Supporting anchors:

- `D:\Code\DiffAudit\Research\workspaces\cross-box\runs\crossbox-pairboard-gsa-targeted-full-overlap-h4-weighted-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\cross-box\runs\crossbox-pairboard-gsa-targeted-full-overlap-20260418-r1\summary.json`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Research/docs/comprehensive-progress.md`: update required
- `Research/docs/future-phase-e-intake.md`: update required
- `Research/docs/mainline-narrative.md`: update required
- `Research/docs/research-autonomous-execution-prompt.md`: update required
- `Research/docs/codex-roadmap-execution-prompt.md`: update required
- `Research/docs/leader-research-ready-summary.md`: update recommended
- `Research/docs/senior-sync-current-difficulties-2026-04-18.md`: update recommended
- `Platform/Runtime`: no direct handoff yet
