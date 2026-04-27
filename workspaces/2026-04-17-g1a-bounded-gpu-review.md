# G1-A Bounded GPU Review

> Status: draft scaffold
> Owner: `Researcher`
> Goal: decide whether `G1-A larger shared-surface tri-score rerun` deserves a bounded GPU slot after `X-89`
> Decision states: `fire-bounded-gpu-review` | `hold-review-only` | `no-go`

## 1. Current truth

- `X-88` selected `G1-A = gray-box tri-evidence audit scorer` as the next honest distinct-family candidate.
- `X-89` landed the CPU gate honestly on frozen `gpu256_undefended / gpu256_defended` surfaces.
- No direct GPU fire is approved yet.
- The next step is review-quality gating, not auto-continuation from the CPU canary.

Canonical evidence:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/x88-cdi-tmiadm-triscore-canary-20260417-175249/audit_summary.json`
- `<DIFFAUDIT_ROOT>/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`

## 2. Shared-Surface Identity

**Status**: FROZEN

The larger rerun would reuse exactly the frozen surfaces from X-89 CPU gate:

**Surfaces in scope**:
- `gpu256_undefended` (PIA + TMIA-DM long_window, 256 samples/split)
- `gpu256_defended` (PIA dropout + TMIA-DM temporal-striding, 256 samples/split)

**Frozen packet artifacts**:
- PIA scores: `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-256/scores.json` (undefended)
- PIA scores: `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-256/scores.json` (defended)
- TMIA-DM family scores: `workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-256-r2-seed1/family-scores.json` (undefended)
- TMIA-DM family scores: `workspaces/gray-box/runs/tmiadm-cifar10-late-window-temporal-striding-defense-20260416-gpu-256-r2-seed1/family-scores.json` (defended)

**Frozen contract fields** (from X-89 identity alignment):
- `dataset_root`: `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/assets/pia/datasets`
- `member_split_root`: `<DIFFAUDIT_ROOT>/Research/external/PIA/DDPM`
- `model_dir`: `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/assets/pia/checkpoints/cifar10_ddpm`
- `max_samples`: 256
- `num_samples`: 256
- `sample_count_per_split`: 256
- `tmiadm_index_strategy`: `adopt-pia-order`

**Identity drift definition**: Any change to the above artifact paths, runtime fields, or index alignment strategy constitutes drift and triggers kill gate.

## 3. Host-Fit Budget

**Status**: APPROVED (CPU-only, no GPU required)

**Expected runtime**: 2-4 hours (offline tri-score evaluation on frozen artifacts)

**Resource requirements**:
- VRAM: 0 GB (no GPU inference required)
- RAM: ~4-8 GB (loading frozen score artifacts + numpy operations)
- Disk I/O: Read-only access to existing PIA/TMIA-DM score files

**Likely failure modes**:
- Artifact path mismatch (mitigated by frozen identity checks)
- Index alignment failure (mitigated by `adopt-pia-order` strategy)
- Scorer implementation bug (mitigated by X-89 canary validation)

**Machine blocking**: Minimal. CPU-bound offline evaluation does not block GPU for other work. Can run in background.

**Verdict**: Fits host budget honestly. No thermal throttle risk, no OOM risk, no extended GPU lock.

## 4. Story Delta Vs X-89

**Status**: STORY-CHANGING

**Project-level claim strengthened**:
If the larger shared-surface rerun succeeds, it establishes that **gray-box evidence aggregation across distinct methods (PIA + TMIA-DM) produces actionable audit signals beyond single-method baselines**, even under defense.

**New honesty boundary**:
- X-89 canary (256 samples): AUC 0.854515, TPR@1%FPR 0.130859 (undefended)
- X-89 canary (256 samples): AUC 0.837601, TPR@1%FPR 0.097656 (defended)
- Larger rerun would test whether tri-score advantage holds at **512+ samples** and whether it beats **both** component methods on low-FPR targets

**Why this is not just metric polishing**:
- X-89 proved the tri-score *can* beat baselines on frozen 256-sample surfaces
- Larger rerun tests whether the advantage is **robust to scale** and **generalizes beyond the canary**
- If positive: promotes G1-A from “internal canary” to “bounded gray-box auxiliary evidence”
- If negative: falsifies the tri-score hypothesis and closes the G1-A lane

**Verdict**: Story-changing. Success changes gray-box narrative from “PIA is strongest” to “tri-evidence aggregation is strongest”. Failure closes G1-A and redirects to other challengers.

## 5. Kill Gate

**Status**: FROZEN

**No-fire conditions** (any one triggers kill):

1. **Identity drift**: Artifact paths, runtime fields, or index alignment strategy cannot remain frozen from X-89 canary
2. **Baseline failure**: Larger rerun does not beat **both** zscore_sum baseline (AUC 0.872986 undefended, 0.83287 defended) **and** control_z_linear baseline on at least one low-FPR target (TPR@1%FPR or TPR@0.1%FPR)
3. **AUC collapse**: Tri-score AUC drops >0.005 below X-89 canary AUC on either surface
4. **Contract violation**: Four-metric reporting (AUC/ASR/TPR@1%FPR/TPR@0.1%FPR) cannot be produced, or component metrics (pia/tmiadm/zscore_sum) are missing
5. **Same-family drift**: Rerun becomes another threshold-scan or switching restatement rather than a distinct tri-evidence aggregation

**Explicit fire authorization**: Only if all kill conditions remain false after larger surface evaluation.

## 6. Recommendation

**Decision**: `hold-review-only`

**Rationale**:
- X-89 CPU gate passed honestly with positive canary results
- Shared-surface identity frozen, host-fit budget approved, story delta justified, kill gate written
- **However**: G1-A remains CPU-bound offline evaluation, not a GPU question
- Larger shared-surface rerun (512+ samples) can proceed as **CPU-first bounded review**, not GPU fire
- Only if larger CPU review produces story-changing positive verdict should GPU question be reopened

**Next step**: Execute larger shared-surface tri-score evaluation as bounded CPU review (X-90), not GPU slot. Reserve GPU authorization for genuinely new hypothesis after G1-A verdict lands.

## 7. Evidence Checklist

- [x] shared-surface identity frozen with explicit drift rules
- [x] host-fit budget written with concrete cost/risk
- [x] story delta written in project-level terms
- [x] kill gate written as explicit no-fire conditions
- [x] final recommendation selected: `hold-review-only`

## 8. Changelog Stub

When the review closes, append one new X-series line to `<DIFFAUDIT_ROOT>/Research/ROADMAP.md` with:

- review outcome
- recommendation
- whether GPU is authorized
- why the decision is honest
