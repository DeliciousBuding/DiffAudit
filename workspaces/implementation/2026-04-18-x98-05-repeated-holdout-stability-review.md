# 2026-04-18 X-98 05 Repeated Holdout Stability Review

## Question

Can the first enlarged `05-cross-box` packet survive a real `5x` stratified `50/50` repeated holdout strongly enough to change the current `05` state?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\crossbox_pairboard.py`
- `D:\Code\DiffAudit\Research\src\diffaudit\cli.py`
- `D:\Code\DiffAudit\Research\tests\test_crossbox_pairboard.py`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-cifar10-runtime-mainline-20260416-gpu-2048-cdi-r1\scores.json`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-loss-score-export-targeted-pia2048-overlap-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\cross-box\runs\crossbox-pairboard-gsa-targeted-pia2048-repeated-20260418-r1\summary.json`

## What Landed

### 1. Pairboard now supports repeated holdout directly

`analyze-crossbox-pairboard` now accepts `--repeats`, reuses the same shared packet, and emits:

- per-run held-out candidate metrics
- aggregate `mean / std / min / max`
- per-candidate comparison against transferred `best_single`

So the old `05` blocker is no longer “we need a separate evaluator script”, but purely whether the current enlarged packet is honest enough.

### 2. One real `5x` repeated-holdout board is now landed on the enlarged packet

Using:

- `PIA 2048` score packet
- targeted `GSA loss-score export`

the shared pairboard remains:

- shared member count = `45`
- shared nonmember count = `35`

and now has a real repeated-holdout summary on top of it.

## Actual Read

### Repeat-level selection truth

- `best_single` selected surface = `pia` in `5/5` repeats

### Aggregate test-board read

Mean `AUC` across `5` held-out repeats:

- `best_single = 0.768287`
- `weighted_average = 0.781586`
- `logistic_2feature = 0.781074`
- `support_disconfirm_neutral = 0.787212`

Mean `TPR@1%FPR` across `5` held-out repeats:

- `best_single = 0.130435`
- `weighted_average = 0.243478`
- `logistic_2feature = 0.278261`
- `support_disconfirm_neutral = 0.226087`

Per-repeat comparison against `best_single`:

- `weighted_average`: `AUC` win `4/5`, `TPR@1%FPR` win `4/5`, mean tail lift `+0.113043`
- `logistic_2feature`: `AUC` win `4/5`, `TPR@1%FPR` win `4/5`, mean tail lift `+0.147826`
- `support_disconfirm_neutral`: `AUC` win `4/5`, `TPR@1%FPR` win `4/5`, mean tail lift `+0.095652`

### Honest boundary

This is stronger than the old one-split read, but still bounded:

- each fusion candidate still has `1/5` held-out split that regresses on `AUC` and low-FPR tail relative to `best_single`
- `ASR` does not improve; it is flat or worse across the fusion candidates
- the held-out board is still only `23 member / 17 nonmember` per repeat, so the `1% FPR` field remains coarse

So this is not yet release-grade low-FPR truth.

## Verdict

- `x98_05_repeated_holdout_stability_verdict = positive but bounded`

More precise reading:

1. `05` now has real repeated-holdout evidence, not just one promising split
2. the `GSA + PIA` fusion story survives most repeats on `AUC` and low-FPR tail
3. current packet is still too small to promote `05` into a hard `go` or `H4` transition
4. the next honest gate is larger matched-packet expansion or finer tail-board confirmation, not another wording-only upgrade

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `near-term active slot = 05-cross-box`
- `current 05 state = repeated-holdout positive but bounded; larger-packet confirmation still required`
- `04-defense` may continue as the next bounded scouting slot, but it should not overwrite the current `05` boundary

## Canonical Evidence Anchor

Primary anchor:

- `D:\Code\DiffAudit\Research\workspaces\cross-box\runs\crossbox-pairboard-gsa-targeted-pia2048-repeated-20260418-r1\summary.json`

Supporting anchors:

- `D:\Code\DiffAudit\Research\workspaces\cross-box\runs\crossbox-pairboard-gsa-targeted-pia2048-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-loss-score-export-targeted-pia2048-overlap-20260418-r1\summary.json`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Research/docs/comprehensive-progress.md`: update required
- `Research/docs/future-phase-e-intake.md`: update required
- `Research/docs/mainline-narrative.md`: update required
- `Research/docs/research-autonomous-execution-prompt.md`: update required
- `Research/docs/codex-roadmap-execution-prompt.md`: update required
- `Platform/Runtime`: no direct handoff yet
