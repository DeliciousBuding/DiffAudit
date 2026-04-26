# 2026-04-18 X-99 05 Full Overlap Pairboard And H4 Gate Review

## Question

After `X-98`, can `05-cross-box` move beyond the small `45 / 35` enlarged packet by exporting the true same-label `GSA target x PIA split` overlap and deciding whether `H4 tail-gated cascade` is now honestly unlocked?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\pia_adapter.py`
- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\crossbox_pairboard.py`
- `D:\Code\DiffAudit\Research\src\diffaudit\cli.py`
- `D:\Code\DiffAudit\Research\tests\test_pia_adapter.py`
- `D:\Code\DiffAudit\Research\tests\test_crossbox_pairboard.py`
- `D:\Code\DiffAudit\Research\workspaces\cross-box\materials\05-gsa-pia-full-overlap-20260418-r1\`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-packet-score-export-gsa-full-overlap-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-loss-score-export-targeted-full-overlap-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\cross-box\runs\crossbox-pairboard-gsa-targeted-full-overlap-20260418-r1\summary.json`

## What Landed

### 1. `PIA packet export` now supports exact-index files

The repo can now export one CPU-first `PIA` packet on explicit member/nonmember index lists and emits a pairboard-ready `scores.json`.

This matters because the old `05` ceiling was no longer the fusion evaluator itself, but the lack of a way to score the real same-label overlap outside one frozen `2048` runtime subset.

### 2. Real same-label overlap is much larger than the old `45 / 35` board

Static overlap audit on current assets showed:

- unique `GSA target-member ∩ PIA member` = `461`
- unique `GSA target-nonmember ∩ PIA nonmember` = `474`
- union allowlist for targeted `GSA` export = `935`

So the old `45 / 35` board was not the real overlap ceiling; it was only the ceiling of the previously frozen `PIA 2048` subset.

### 3. One true enlarged full-overlap pairboard now exists

The new CPU-first packet path produced:

- `PIA exact packet = 461 member / 474 nonmember`
- `GSA targeted full-overlap export = 522 member / 523 nonmember`
- aligned shared pairboard = `461 member / 474 nonmember`

## Actual Read

### Single held-out board

On the primary held-out split:

- `best_single AUC = 0.836539`
- `weighted_average AUC = 0.826675`
- `logistic_2feature AUC = 0.840156`
- `best_single TPR@1%FPR = 0.125541`
- `weighted_average TPR@1%FPR = 0.134199`
- `logistic_2feature TPR@1%FPR = 0.168831`
- `best_single TPR@0.1%FPR = 0.004329`
- `weighted_average TPR@0.1%FPR = 0.090909`
- `logistic_2feature TPR@0.1%FPR = 0.060606`

### `5x` repeated holdout aggregate

Mean held-out metrics:

- `best_single`: `AUC = 0.810430`, `ASR = 0.751282`, `TPR@1%FPR = 0.096104`, `TPR@0.1%FPR = 0.007792`
- `weighted_average`: `AUC = 0.806579`, `ASR = 0.736325`, `TPR@1%FPR = 0.137662`, `TPR@0.1%FPR = 0.075325`
- `logistic_2feature`: `AUC = 0.815292`, `ASR = 0.750000`, `TPR@1%FPR = 0.148918`, `TPR@0.1%FPR = 0.046753`

Comparison against transferred `best_single`:

- `weighted_average`:
  - `AUC`: win `2/5`, loss `3/5`
  - `TPR@1%FPR`: win `5/5`
  - `TPR@0.1%FPR`: win `5/5`
- `logistic_2feature`:
  - `AUC`: win `4/5`, loss `1/5`
  - `ASR`: win `3/5`, tie `1/5`, loss `1/5`
  - `TPR@1%FPR`: win `5/5`
  - `TPR@0.1%FPR`: win `5/5`

### Honest reading

The promoted candidate is now clear:

- `weighted_average` still looks like a tail-only auxiliary fusion because it loses `AUC` and `ASR`
- `logistic_2feature` is the first candidate that keeps the board honest enough to matter: it improves `AUC`, keeps `ASR` near-flat, and wins both low-FPR tails in every repeated split

This is materially stronger than `X-98`.

## Verdict

- `x99_05_full_overlap_pairboard_verdict = positive`

More precise reading:

1. the larger same-label overlap is now real, not inferred
2. `05-H1/H2` has now crossed the stronger gate that `X-98` could not yet clear
3. `logistic_2feature` is now the honest promoted fusion candidate inside `05`
4. current evidence is strong enough to allow bounded entry into `H4 tail-gated cascade`
5. this still does **not** justify a generic cross-box scalar or a paper-level final headline outside the bounded `05` contract

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `near-term active slot = 05-cross-box`
- `current 05 state = stable low-FPR tail-lift confirmed on enlarged matched packet`
- `current promoted 05 candidate = logistic_2feature`
- `next honest 05 move = bounded H4 tail-gated cascade`

## Canonical Evidence Anchor

Primary anchor:

- `D:\Code\DiffAudit\Research\workspaces\cross-box\runs\crossbox-pairboard-gsa-targeted-full-overlap-20260418-r1\summary.json`

Supporting anchors:

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-packet-score-export-gsa-full-overlap-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-loss-score-export-targeted-full-overlap-20260418-r1\summary.json`

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
