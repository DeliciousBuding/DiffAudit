# 2026-04-17 Cross-Permission CPU Canary Interpretation

## Question

Do the new white-box and gray-box CPU canaries already form one honest joint bridge packet strong enough to restore a bounded GPU candidate, or are they still only two separate scaffolds?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-cross-permission-cpu-scaffold-implementation.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-cross-permission-support-contract.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\cross-permission-masked-packet-canary-20260417-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\cross-permission-masked-packet-canary-20260417-r1\records.jsonl`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-packet-score-export-20260417-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-packet-score-export-20260417-r1\sample_scores.jsonl`
- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\gsa_observability.py`
- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\pia_adapter.py`

## What The Two Canaries Actually Prove

### White-box canary

The white-box canary proves:

1. the repo can export one fixed member/control activation pair
2. it can derive a channel mask from that pair
3. it can write pre-mask and post-mask tensor artifacts

Observed packet:

- member sample:
  - `target-member/00-data_batch_1-00965.png`
- control sample:
  - `target-nonmember/00-data_batch_1-00467.png`
- `selected_delta_retention_ratio = 0.5`

### Gray-box canary

The gray-box canary proves:

1. the repo can export packet-local `PIA` scores
2. it can write per-sample score rows
3. it can compute one `member_control_score_gap`

Observed packet:

- member indices:
  - `10365 / 25853 / 41222 / 7409`
- non-member indices:
  - `6640 / 7423 / 45313 / 9057`
- `member_control_score_gap = 18.217201`

## Why This Is Still Not One Joint Packet

Two blockers remain.

### 1. Packet identity mismatch

The white-box packet and gray-box packet are not the same packet.

White-box packet identity is frozen as:

- one named canary sample
- one named control sample

Gray-box packet identity is currently frozen as:

- four member split indices
- four non-member split indices

The repo does not yet show that:

1. the gray-box packet contains the same member/control objects as the white-box packet
2. the packet sizes are intentionally reconciled
3. the same-sample bridge semantics are frozen

So current canaries are:

- `same hypothesis family`
- not `same packet`

### 2. White-box intervention is still offline-tensor masking

The current white-box canary applies the mask on captured activation tensors after export.

That is useful for scaffold validation.

But it is still not the same thing as:

- intervening inside the forward path and then measuring downstream cross-surface effect

So the current white-box canary proves:

- packet-local manipulation surface exists

It does not yet prove:

- a bridge-ready in-model intervention packet exists

## GPU Re-Release Decision

Because of those two blockers, the current canaries do **not** justify restoring the GPU candidate.

Current honest reading:

- the implementation blocker is cleared
- the packet-assembly blocker is not

Therefore:

- `active_gpu_question = none`
- `next_gpu_candidate = none`

## Why This Is Useful Anyway

The canaries still materially improve repo truth because they show:

1. the missing surfaces from `I-C.4` were real and are now solved
2. the next blocker is no longer implementation vagueness
3. the next blocker is now precise packet assembly and intervention semantics

That is a real research advance, even though it is still below GPU release.

## Verdict

- `cross_permission_cpu_canary_interpretation_verdict = blocked`

More precise reading:

1. `I-C.7` is now satisfied:
   - the CPU canaries have been interpreted
2. GPU re-release decision:
   - still `none`
3. remaining blockers:
   - same-packet identity is not yet frozen across white-box and gray-box
   - white-box intervention is still offline tensor masking rather than in-model bridge execution

## Next Step

- `next_live_cpu_first_lane = I-C.8 same-packet identity and in-model intervention contract`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: light sync required
- `workspaces/white-box/plan.md`: update required
- `workspaces/implementation/challenger-queue.md`: light sync required
- `Leader/materials`: no sync needed
- `Platform/Runtime`: no direct handoff; still below joint-packet truth
