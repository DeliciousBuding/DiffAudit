# 2026-04-18 X-97 05 Targeted GSA PIA2048 Shared Packet Expansion

## Question

Can `05-cross-box` move beyond tiny overlap infrastructure validation by exporting one larger matched `GSA` packet against an already-landed `PIA 2048` surface?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\gsa.py`
- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\crossbox_pairboard.py`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-cifar10-runtime-mainline-20260416-gpu-2048-cdi-r1\scores.json`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-loss-score-export-targeted-pia2048-overlap-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\cross-box\runs\crossbox-pairboard-gsa-targeted-pia2048-20260418-r1\summary.json`

## What Landed

### 1. `GSA loss-score export` now supports targeted sample-ID filtering

The export surface now accepts one allowlist and only exports matching sample IDs from each split.

This matters because the old `05` blocker was no longer “no white-box scalar surface”, but “no large enough matched shared packet”.

### 2. First targeted overlap export succeeded on real assets

Using the full `PIA 2048` member/nonmember ID set as the allowlist, the targeted `GSA` export produced:

- target member sample count = `89`
- target non-member sample count = `77`

This is already a major step up from the earlier bounded actual packet that only gave a tiny overlap read.

### 3. First expanded actual pairboard now exists

Running the generic pairboard on:

- `PIA 2048`
- targeted `GSA loss-score export`

produced one actual shared subset of:

- shared member count = `45`
- shared nonmember count = `35`

This is the first `05` packet that is large enough to count as more than infrastructure smoke.

## Actual Read

The packet is promising but still bounded.

Observed held-out test board:

- `best_single AUC = 0.874680`
- `weighted_average AUC = 0.864450`
- `logistic_2feature AUC = 0.890026`
- `support_disconfirm_neutral AUC = 0.808184`

Observed held-out low-FPR fields:

- `best_single TPR@1%FPR = 0.043478`
- `weighted_average TPR@1%FPR = 0.434783`
- `logistic_2feature TPR@1%FPR = 0.304348`
- `support_disconfirm_neutral TPR@1%FPR = 0.391304`

But the honest boundary is still tight:

- the held-out test split is only `23 member / 17 nonmember`
- so `1% FPR` is still a coarse zero-false-positive operating point
- only one holdout split has been read so far

So this packet is **not** release-grade low-FPR truth yet.

## Verdict

- `x97_05_targeted_gsa_pia2048_shared_packet_expansion_verdict = positive but bounded`

More precise reading:

1. `05` now has a first genuinely larger matched packet
2. fusion candidates are directionally interesting on the held-out board
3. current low-FPR fields are still too coarse for promotion
4. the next honest move is repeated-holdout stability on this enlarged packet or another matched-packet expansion, not immediate go-claim wording

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `near-term active slot = 05-cross-box`
- `current 05 state = first larger matched packet landed; stability review pending`

## Canonical Evidence Anchor

Primary anchor:

- `D:\Code\DiffAudit\Research\workspaces\cross-box\runs\crossbox-pairboard-gsa-targeted-pia2048-20260418-r1\summary.json`

Supporting anchors:

- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-loss-score-export-targeted-pia2048-overlap-20260418-r1\summary.json`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Research/docs/comprehensive-progress.md`: update required
- `Research/docs/future-phase-e-intake.md`: update required
- `Research/docs/mainline-narrative.md`: update required
- `Platform/Runtime`: no direct handoff yet
