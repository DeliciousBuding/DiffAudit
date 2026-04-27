# 2026-04-21 X-125 R2-5b PIA + SimA Support-Fusion Bounded Review

## Question

After `X-124` closed the `SimA` packet-export gap, does `PIA + SimA` now deliver an honest low-FPR support-fusion gain over `PIA` best-single on a real shared packet?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-21-x124-r25c-sima-packet-score-export-contract-landing.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-packet-score-export-gsa-full-overlap-20260418-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-packet-score-export-gsa-full-overlap-20260418-r1/scores.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/sima-cifar10-runtime-rescan-20260416-cpu-32-r2/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/sima-packet-score-export-pia-full-overlap-20260421-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/sima-packet-score-export-pia-full-overlap-20260421-r1/scores.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/cross-box/runs/crossbox-pairboard-pia-sima-full-overlap-20260421-r1/summary.json`

## Bounded Packet Used

- frozen `PIA` packet:
  - `461 member / 474 nonmember`
  - source: `pia-packet-score-export-gsa-full-overlap-20260418-r1`
- `SimA` export:
  - same exact member/nonmember indices
  - `timestep = 160`
  - `p_norm = 4`
  - source: `sima-packet-score-export-pia-full-overlap-20260421-r1`
- fusion evaluator:
  - `analyze-crossbox-pairboard`
  - `5x` repeated holdout
  - candidates:
    - `best_single`
    - `weighted_average`
    - `logistic_2feature`
    - `support_disconfirm_neutral`

## Findings

### 1. `PIA` remains the best single surface

Across repeated holdout:

- selected best single surface = `pia` in `5 / 5` runs

So this review does **not** change the base gray-box headline.

### 2. `logistic_2feature` gives real bounded uplift on headline smooth metrics

Repeated-holdout aggregate:

- `best_single auc mean = 0.818697`
- `logistic_2feature auc mean = 0.833332`
- `best_single asr mean = 0.752564`
- `logistic_2feature asr mean = 0.771368`

Win/loss against `best_single`:

- `auc`: `5 / 5` wins
- `asr`: `5 / 5` wins

So `PIA + SimA` is not empty; it adds bounded fused signal on the shared packet.

### 3. Low-FPR improvement is only partial, not strong enough for promotion

Repeated-holdout aggregate:

- `best_single tpr@1%fpr mean = 0.100433`
- `logistic_2feature tpr@1%fpr mean = 0.108225`
- `best_single tpr@0.1%fpr mean = 0.043290`
- `logistic_2feature tpr@0.1%fpr mean = 0.041558`

Win/loss against `best_single`:

- `tpr@1%fpr`: `4 wins / 1 loss`
- `tpr@0.1%fpr`: `2 wins / 1 tie / 2 losses`

So the current fused board shows:

- partial `1%`-FPR tail help
- no stable `0.1%`-FPR lift

That is below the threshold for promoting this line into a stronger low-FPR claim.

### 4. `weighted_average` is weaker as a tail story

`weighted_average` also improves `AUC / ASR`, but its repeated-holdout tail comparison is worse:

- `tpr@1%fpr mean_delta = -0.004329`
- `tpr@0.1%fpr mean_delta = -0.006061`

So the honest bounded fused candidate is `logistic_2feature`, not `weighted_average`.

## Verdict

`positive but bounded`.

Sharper control truth:

1. `PIA + SimA` support-fusion is now really executable and non-empty
2. the best bounded fused candidate is `logistic_2feature`
3. it improves `AUC / ASR` and partially helps `TPR@1%FPR`
4. it does **not** give a stable `TPR@0.1%FPR` lift
5. therefore this line stays an auxiliary gray-box sidecar, not a promoted mainline change and not a new GPU question
6. gray-box should now yield the next `CPU-first` slot back to non-graybox reselection / `I-A` / system-consumable sync

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-21-x125-r25b-pia-sima-support-fusion-bounded-review.md`

## Handoff

- `Research/ROADMAP.md`: yes
- `docs/comprehensive-progress.md`: yes
- `docs/research-autonomous-execution-prompt.md`: yes
- `docs/codex-roadmap-execution-prompt.md`: yes
- `workspaces/implementation/challenger-queue.md`: yes
- `Platform/Runtime`: no

Reason:

This review changes only research-side lane ordering and interpretation. It does not justify any Runtime field, Platform schema, or admitted-table change.
