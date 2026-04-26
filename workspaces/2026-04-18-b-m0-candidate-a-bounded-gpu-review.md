# B-M0 Candidate A: Bounded Shadow-LR White-Box Loss-Score Follow-Up

> Date: 2026-04-18
> Owner: Researcher
> Context: Root ROADMAP §4 B-M0 (2026-04-17 override) - bounded GPU release candidate

## Question

Should bounded shadow-LR / likelihood-ratio follow-up on frozen X-75/X-77 white-box loss-score packet receive GPU authorization?

## 1. Shared-Surface Identity

**Status**: FROZEN

The follow-up reuses exactly the frozen X-75/X-77 packet identity:

**Frozen packet contract** (X-75/X-77):
- Same admitted `DDPM/CIFAR10` target/shadow asset family
- `extraction_max_samples = 64` per split
- Exported scalar loss-score artifacts
- Shadow-oriented threshold-transfer baseline (X-77)

**Frozen assets**:
- Target: `gsa-cifar10-1k-3shadow-epoch300-rerun1` (admitted)
- Shadows: same 3-shadow family
- Export artifacts: `gsa-loss-score-export-bounded-actual-20260418-r1`

**X-77 baseline metrics** (shadow-threshold-transfer):
- `AUC = 0.699463`
- `ASR = 0.632812`
- `TPR@1%FPR = 0.03125`
- `TPR@0.1%FPR = 0.03125`

**New hypothesis**: Shadow-distribution likelihood-ratio (LR) transfer
- Same packet identity (frozen)
- Same extraction artifacts (reused)
- New scoring layer: LR-based transfer instead of threshold-based
- Hypothesis: LR-transfer beats X-77 threshold-transfer on low-FPR targets

**Identity drift definition**: Any change to frozen packet identity, asset paths, or extraction contract triggers kill gate.

## 2. Host-Fit Budget

**Status**: APPROVED (CPU-only, no GPU required)

**Expected runtime**: 2-4 hours (shadow distribution construction + LR evaluation)

**Resource requirements**:
- VRAM: 0 GB (no GPU inference, reuses frozen export artifacts)
- RAM: ~4-8 GB (shadow distribution construction on CIFAR10)
- Disk I/O: Read-only access to frozen loss-score artifacts

**Shadow distribution construction**:
- CIFAR10 scale: 64 samples per split × 3 shadows = 192 member + 192 nonmember scores
- Fits 4070 Laptop 8GB without GPU (CPU-only numpy operations)
- No new model training or shadow model inference required

**Likely failure modes**:
- LR estimation unstable at bounded scale (64 samples) - mitigated by shadow pooling
- LR-transfer collapses to threshold-transfer - still produces valid negative verdict
- Numerical instability in likelihood computation - mitigated by log-space operations

**Machine blocking**: Minimal. CPU-bound offline evaluation, no GPU lock, can run in background.

**Verdict**: Fits host budget honestly. No thermal throttle risk, no OOM risk, no extended GPU lock.

## 3. Story Delta Vs X-77

**Status**: STORY-CHANGING

**Project-level claim strengthened**:
If shadow-LR transfer succeeds, it establishes that **white-box loss-feature attacks support distinct scoring methods beyond threshold-transfer**, providing auxiliary evidence that loss-score signals are robust to scorer choice.

**New honesty boundary**:
- X-77 baseline (threshold-transfer): `AUC 0.699463 / TPR@1%FPR 0.03125`
- Shadow-LR hypothesis: LR-transfer beats threshold on low-FPR targets
- Success criterion: `TPR@1%FPR` or `TPR@0.1%FPR` improvement over X-77 baseline

**Why this is not just metric polishing**:
- X-77 proved threshold-transfer works on frozen packet
- Shadow-LR tests whether distinct scoring method (LR vs threshold) improves low-FPR
- If positive: promotes white-box loss-feature from "single-scorer auxiliary" to "multi-scorer auxiliary"
- If negative: falsifies LR hypothesis, confirms threshold-transfer as strongest bounded method

**Verdict**: Story-changing. Success changes white-box narrative from "threshold-only auxiliary" to "multi-scorer auxiliary". Failure confirms threshold-transfer as strongest bounded method.

## 4. Kill Gate

**Status**: FROZEN

**No-fire conditions** (any one triggers kill):

1. **Identity drift**: Frozen packet identity, asset paths, or extraction contract cannot remain frozen from X-75/X-77
2. **Baseline failure**: LR-transfer does not beat X-77 threshold-transfer baseline on at least one low-FPR target (`TPR@1%FPR` or `TPR@0.1%FPR`)
3. **AUC collapse**: LR-transfer AUC drops >0.01 below X-77 baseline AUC (0.699463)
4. **Contract violation**: Four-metric reporting (`AUC/ASR/TPR@1%FPR/TPR@0.1%FPR`) cannot be produced
5. **Same-family drift**: Follow-up becomes another threshold-scan restatement rather than distinct LR-based scoring

**Explicit fire authorization**: Only if all kill conditions remain false after CPU-first LR evaluation.

## 5. Recommendation

**Decision**: `hold-review-only`

**Rationale**:
- X-75/X-77 frozen packet provides same-asset identity for LR follow-up
- Shadow-LR hypothesis is genuinely new (distinct from X-77 threshold-transfer)
- Host-fit budget approved (CPU-only, 2-4h, no GPU required)
- Story delta justified (multi-scorer auxiliary evidence vs single-scorer)
- Kill gate written with explicit no-fire conditions
- **However**: B-M0 Candidate A remains CPU-bound offline evaluation, not a GPU question
- Shadow distribution construction and LR evaluation can proceed as **CPU-first bounded review**, not GPU fire
- Only if CPU review produces story-changing positive verdict should GPU question be reopened

**Next step**: Execute shadow-LR CPU-first bounded review, not GPU slot. Reserve GPU authorization for genuinely new hypothesis after B-M0 Candidate A verdict lands.

## 6. Evidence Checklist

- [x] shared-surface identity frozen with explicit drift rules
- [x] host-fit budget written with concrete cost/risk
- [x] story delta written in project-level terms
- [x] kill gate written as explicit no-fire conditions
- [x] final recommendation selected: `hold-review-only`

## 7. Changelog Stub

When the review closes, append one new X-series line to `D:/Code/DiffAudit/Research/ROADMAP.md` with:

- review outcome
- recommendation
- whether GPU is authorized
- why the decision is honest
