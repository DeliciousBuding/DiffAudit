# 2026-04-18 X-104 04 H1 Full-Split First Review And Scan Fix

## Question

After `X-103` and the retained-companion read, what does the first target-wide full-split review say about `04-H1`, and is the no-allowlist review path actually executable on the current repo code?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/gsa.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_gsa_adapter.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/defenses/risk_targeted_unlearning.py`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-pilot-k32-20260418-r2/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-k32-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-retained-k32-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-fullsplit-k32-20260418-r1/summary.json`

## What Landed

### 1. The full-split review path needed one real scan fix

The first no-allowlist full-split rerun did not fail on model logic.

It failed because `GSA` dataset scanning still treated every file under the dataset root as a candidate sample.

On the real admitted target datasets, that meant `dataset.json` was passed into `PIL.Image.open`, which raised `UnidentifiedImageError`.

The repo now fixes that path by:

1. making `_iter_dataset_files` admit only image-like suffixes
2. making `_dataset_has_files` follow the same image-only rule
3. adding a regression test that keeps `dataset.json` and `notes.txt` out of the extracted dataset file list

This is a real execution-path fix, not just a test-only patch: the old subset reviews avoided it only because the sample-ID allowlist filtered out non-image files before extraction.

### 2. One first full-split review now exists

The repo now contains a target-wide `borrowed-shadow / defense-unaware threshold-transfer` review on:

- `1000` target members
- `1000` target nonmembers
- baseline target checkpoint
- defended `k32` retain+forget checkpoint

So `04-H1` is no longer missing its first target-wide read.

## Actual Read

### Full-split baseline vs defended target-transfer metrics

- baseline:
  - `AUC = 0.618043`
  - `ASR = 0.5515`
  - `TPR@1%FPR = 0.018`
  - `TPR@0.1%FPR = 0.006`
- defended:
  - `AUC = 0.596696`
  - `ASR = 0.5665`
  - `TPR@1%FPR = 0.011`
  - `TPR@0.1%FPR = 0.003`

Transfer deltas (`defended - baseline`):

- `AUC = -0.021347`
- `ASR = +0.015`
- `TPR@1%FPR = -0.007`
- `TPR@0.1%FPR = -0.003`

### Combined reading across the three attached boards

Current attached board stack is now:

1. forgotten subset:
   - clearly negative, especially on low-FPR tails
2. retained high-risk companion subset:
   - mixed but weak
   - `AUC` worsens, while the tail only rebounds slightly
3. full split:
   - also negative
   - milder than the forgotten-subset collapse, but still not a defense-positive read

### Honest reading

This matters because the lane is no longer blocked on "we still need one wider board to know whether the subset view was misleading".

The wider board is now here, and it does not reverse the direction.

The most honest read is:

1. `04-H1` now has a real pilot plus three attached attack-side boards
2. none of those attached reads justify successor-positive wording
3. the strongest available stack is now `forgotten negative + retained mixed/weak + full-split negative`
4. all of it is still `borrowed-shadow / defense-unaware`, so this is still not the final family verdict

## Verdict

- `x104_04_h1_fullsplit_first_review_verdict = negative but still defense-unaware`

More precise reading:

1. `04-H1` is no longer only "selection complete + first pilot exists"
2. it is now "first pilot exists + target-wide first read exists"
3. but that target-wide first read is still unfavorable

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `04 current state = actual pilot exists + forgotten negative + retained mixed/weak + full-split negative`
- `next honest move = sync higher-layer wording, then decide whether defense-aware rerun is worth the cost before opening any heavier fallback family`

## Canonical Evidence Anchor

Primary anchor:

- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-fullsplit-k32-20260418-r1/summary.json`

Supporting anchors:

- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-k32-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-retained-k32-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/gsa.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_gsa_adapter.py`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Research/docs/comprehensive-progress.md`: update required
- `Research/docs/mainline-narrative.md`: update required
- `Research/docs/future-phase-e-intake.md`: update required
- `Platform/Runtime`: still no direct handoff yet
