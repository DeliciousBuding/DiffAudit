# 2026-04-18 X-75 White-Box Bounded Loss-Score First Packet Selection After X-74 Implementation

## Question

Now that bounded same-asset loss-score export is real, what is the first honest packet/evaluation contract for this white-box loss-feature lane?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-18-x74-whitebox-bounded-internal-loss-score-export-implementation-after-x73-surface-review.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-loss-score-export-bounded-smoke-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\pia_adapter.py`
- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\recon.py`
- `D:\Code\DiffAudit\Research\src\diffaudit\defenses\dpdm_w1.py`
- `D:\Code\DiffAudit\Research\docs\paper-reports\ocr\white-box\2025-popets-white-box-membership-inference-diffusion-models\pages\page-0016.md`
- `D:\Code\DiffAudit\Research\docs\paper-reports\ocr\white-box\2025-popets-white-box-membership-inference-diffusion-models\pages\page-0017.md`

## Packet Selection Review

### 1. The first packet should stay threshold-style, not classifier-heavy

The current goal is to land the first honest same-asset loss-feature packet, not to introduce a new learned attack family.

So the first packet should **not** be:

- `LiRA / Strong LiRA`
- logistic-regression or XGBoost-first loss-score promotion
- a mixed score+gradient fusion board

The honest first packet is the lowest-complexity reading closest to the current `LSA*`-style contract:

- scalar loss scores
- threshold-style evaluation
- shadow-to-target transfer

### 2. Orientation must be frozen from the shadow side, not guessed from target

The bounded smoke already shows that raw loss-score direction is not safe to assume from one target pair alone.

So the first packet should not hardcode:

- `member-higher`
- or `member-lower`

Instead it should:

1. pool all bounded shadow-member scores
2. pool all bounded shadow-nonmember scores
3. infer one score orientation from the shadow side
4. derive one shadow threshold under that frozen orientation
5. evaluate the target side with the same orientation and threshold

This keeps the packet honest to the threat-model split and avoids target-side hindsight selection.

### 3. The first bounded budget should freeze to `64` per split

The best current bounded first-packet budget is:

- `extraction_max_samples = 64` per target/shadow split

Why this is the honest first budget:

- it matches the repository's existing bounded white-box packet scale
- it is materially stronger than the `1`-sample smoke
- it stays CPU-first and host-fit
- it is still clearly below any expensive `LiRA`-style release

### 4. Low-FPR must be reported, but this packet is still below low-FPR release

The first packet should still report:

- `AUC`
- `ASR`
- `TPR@1%FPR`
- `TPR@0.1%FPR`

But the honest boundary is:

- with `64` non-member target samples, low-FPR reading is still coarse
- so the first packet is useful for contract validation and early direction
- not for claiming a release-grade low-FPR white-box loss-feature result

### 5. The first packet contract

Freeze the first packet as:

- same admitted `DDPM/CIFAR10` target/shadow asset family
- bounded `extraction_max_samples = 64`
- exported scalar loss-score artifacts only
- pooled shadow member/non-member orientation review
- shadow-derived threshold transfer onto target
- target self-board retained as diagnostic only

## Verdict

- `x75_whitebox_bounded_loss_score_first_packet_selection_verdict = positive but bounded`

More precise reading:

1. the first honest packet is now selected
2. it is `threshold-style`, not classifier-heavy
3. it is `shadow-oriented + shadow-threshold-transferred`, not target-retuned
4. it is bounded at `64` per split
5. it remains below release-grade low-FPR honesty

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `current execution lane = X-76 white-box bounded loss-score threshold evaluator implementation after X-75 packet selection`
- `current CPU sidecar = I-A higher-layer boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Research/workspaces/implementation/challenger-queue.md`: update required
- `Research/workspaces/white-box/plan.md`: update required
- `Research/docs/comprehensive-progress.md`: update required
- `Research/docs/mainline-narrative.md`: update required
- root `ROADMAP.md`: update required
- prompt/bootstrap docs: update required
- `Platform/Runtime`: no direct handoff yet
- future handoff trigger: if `X-76` changes consumer-visible score packet schema or summary logic, upgrade to note-level system handoff review
- competition/materials sync: note-level only
