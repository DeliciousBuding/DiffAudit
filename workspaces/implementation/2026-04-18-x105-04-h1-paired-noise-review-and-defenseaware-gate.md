# 2026-04-18 X-105 04 H1 Paired-Noise Review And Defense-Aware Gate

## Question

After `X-104`, do the negative `04-H1` readings persist once baseline and defended target exports are forced onto the same per-sample noise draws, and is the current `k32` pilot worth a heavier defense-aware rerun?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/gsa.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/defenses/risk_targeted_unlearning.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/cli.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_gsa_adapter.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_risk_targeted_unlearning.py`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-k32-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-retained-k32-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-fullsplit-k32-pairednoise-20260418-r1/summary.json`

## What Landed

### 1. The review surface now supports paired-noise target export

The repo now exposes one stronger review control for `GSA` loss-score export:

1. `_extract_gsa_loss_scores` accepts `noise_seed`
2. noise is derived deterministically from `noise_seed + sample_key`
3. `review_risk_targeted_unlearning_pilot` forwards that seed to all four target exports
4. the CLI surface now exposes `--noise-seed`

This does **not** make the review defense-aware.

But it removes one obvious target-side fairness problem: baseline and defended are no longer compared under different random noise draws for the same sample.

### 2. Three paired-noise target boards now exist

Using `noise_seed = 0`, the repo now has:

1. `forgotten subset + matched controls`
2. `retained high-risk companion subset`
3. `full target split`

So `04-H1` no longer depends only on the earlier unpaired target reruns.

## Actual Read

### Paired-noise target-transfer metrics

Forgotten subset:

- baseline:
  - `AUC = 0.845679`
  - `ASR = 0.513889`
  - `TPR@1%FPR = 0.25`
  - `TPR@0.1%FPR = 0.25`
- defended:
  - `AUC = 0.827932`
  - `ASR = 0.555556`
  - `TPR@1%FPR = 0.194444`
  - `TPR@0.1%FPR = 0.194444`

Retained companion:

- baseline:
  - `AUC = 0.601307`
  - `ASR = 0.514286`
  - `TPR@1%FPR = 0.083333`
  - `TPR@0.1%FPR = 0.083333`
- defended:
  - `AUC = 0.597222`
  - `ASR = 0.542857`
  - `TPR@1%FPR = 0.083333`
  - `TPR@0.1%FPR = 0.083333`

Full split:

- baseline:
  - `AUC = 0.623331`
  - `ASR = 0.5585`
  - `TPR@1%FPR = 0.027`
  - `TPR@0.1%FPR = 0.002`
- defended:
  - `AUC = 0.617696`
  - `ASR = 0.5805`
  - `TPR@1%FPR = 0.026`
  - `TPR@0.1%FPR = 0.002`

### What changed relative to the old unpaired reads

The same-noise control clearly matters.

It weakens the severity of the earlier negative read, especially on the forgotten subset and on full split.

But it does **not** reverse the sign:

1. forgotten subset stays negative
2. retained subset stays flat-to-weak rather than encouraging
3. full split stays slightly negative / non-positive

So the older target-side read was partly noisier than it should have been, but it was not directionally fabricated by that noise mismatch.

### Paired-noise shift decomposition

On the paired-noise full-split export:

- all members mean loss shift = `+0.00755`
- forgotten members mean loss shift = `+0.007678`
- retained high-risk members mean loss shift = `+0.007011`
- all nonmembers mean loss shift = `+0.007266`
- forgotten matched nonmembers mean loss shift = `+0.006956`
- retained matched nonmembers mean loss shift = `+0.007364`

This is the key mechanistic read:

the defended checkpoint is not showing a strong, clean, concentrated shift on the intended forgotten members.

The shift looks broadly global, with only small local differences, and the full-split separation still does not improve enough to justify a heavier next-stage rerun.

## Verdict

- `x105_04_h1_paired_noise_gate_verdict = negative but clarified`

More precise reading:

1. the old target-side review was too noisy for strict sample-level interpretation
2. the new paired-noise reruns make the review surface more honest
3. under that stronger surface, the current `k32` pilot still fails to turn positive

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `04 current state = current k32 pilot not worth defense-aware rerun`
- `next honest move = freeze current H1 instantiation as negative-but-useful, and only reopen 04 GPU spend if a changed pilot produces a clearly stronger same-noise target-side gate`

Practical gate:

1. do **not** defense-aware-rerun the current `k32` checkpoint
2. if `04` is kept alive, first change the pilot, then recheck with paired-noise forgotten/full-split review
3. only after that should a heavier defense-aware shadow retraining question reopen

## Canonical Evidence Anchor

Primary anchors:

- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-k32-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-retained-k32-pairednoise-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/defense/runs/risk-targeted-unlearning-review-fullsplit-k32-pairednoise-20260418-r1/summary.json`

Supporting anchors:

- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/gsa.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/defenses/risk_targeted_unlearning.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/cli.py`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Research/docs/comprehensive-progress.md`: update required
- `Research/docs/mainline-narrative.md`: update required
- `Research/docs/future-phase-e-intake.md`: update required
- `Platform/Runtime`: still no direct handoff yet
