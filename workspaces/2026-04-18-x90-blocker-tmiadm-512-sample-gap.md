# X-90 Blocker: TMIA-DM 512-Sample Gap

> Date: 2026-04-18
> Status: blocked
> Owner: Researcher

## Context

G1-A bounded GPU review (hold-review-only) recommended X-90 larger shared-surface tri-score evaluation to test whether tri-evidence aggregation advantage holds at 512+ samples.

## Blocker

**TMIA-DM does not have 512-sample runs matching PIA 512-sample surfaces.**

### Available Assets

**PIA 512-sample runs** (exist):
- `pia-cifar10-runtime-mainline-20260409-gpu-512-adaptive` (undefended)
- `pia-cifar10-runtime-mainline-dropout-defense-20260409-gpu-512-allsteps-adaptive` (defended)

**TMIA-DM runs** (only 256-sample):
- `tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-256-r2-seed1` (undefended)
- `tmiadm-cifar10-late-window-temporal-striding-defense-20260416-gpu-256-r2-seed1` (defended)

### Identity Alignment Failure

X-90 script attempted to use PIA 512 + TMIA-DM 256, but `graybox_triscore.py` enforces identity alignment:
- PIA 512: 512 members, 512 nonmembers
- TMIA-DM 256: 256 members, 256 nonmembers
- **Error**: `ValueError: Identity alignment mismatch for gpu512_undefended: member count differs between PIA and TMIA-DM.`

## Resolution Options

### Option A: Run TMIA-DM 512-sample (GPU required)
- Execute TMIA-DM on 512-sample surfaces matching PIA 512 member/nonmember indices
- Estimated cost: ~2-4h GPU time (2 surfaces: undefended + defended)
- Verdict: Unblocks X-90 honestly

### Option B: Downgrade X-90 to 256-sample rerun (CPU-only)
- X-89 already evaluated 256-sample surfaces
- Larger rerun would only verify robustness, not test new scale
- Verdict: Weakens story delta justification from G1-A review

### Option C: Defer X-90 until TMIA-DM 512 assets exist
- Keep G1-A as CPU-bound canary (X-89)
- Wait for TMIA-DM 512 runs before claiming scale robustness
- Verdict: Honest but delays G1-A promotion

## Recommendation

**Option A**: Run TMIA-DM 512-sample to unblock X-90.

**Rationale**:
- G1-A review explicitly justified story delta as testing tri-score robustness at scale
- 512-sample is the minimum honest larger surface (2x X-89 canary)
- TMIA-DM 512 runs are bounded (same protocol as 256, just larger sample count)
- Cost is acceptable (2-4h GPU, not multi-day)

## Next Steps

1. Create TMIA-DM 512-sample run scripts for undefended + defended surfaces
2. Execute TMIA-DM 512 runs (GPU slot)
3. Retry X-90 with matched PIA 512 + TMIA-DM 512 surfaces
4. Update Research ROADMAP with X-90 verdict

## Kill Gate

If TMIA-DM 512 runs fail or produce incompatible artifacts, close G1-A as `blocked-needs-assets` and redirect to other challenger lanes.
