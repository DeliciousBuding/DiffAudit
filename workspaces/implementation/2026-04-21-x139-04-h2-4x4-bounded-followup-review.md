# 2026-04-21 X-139 04-H2 4x4 Bounded Follow-Up Review

## Question

After `X-138` selects one minimal `4 / 4` packet-scale enlargement, does that first bounded follow-up reveal any honest defense signal for `04-H2`, or should the lane now yield?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-21-x138-04-h2-bounded-packet-scale-followup-selection.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\runs\h2-run-defense-pilot-4x4-20260421-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\implementation\runs\h2-review-defense-pilot-4x4-20260421-r1\summary.json`

## Findings

### 1. The minimal `4 / 4` bounded pilot is executable and clean

`run-h2-defense-pilot` lands successfully on the same admitted asset family with:

- `member_count = 4`
- `nonmember_count = 4`
- `device = cpu`
- same frozen checkpoint identity and runtime defaults

### 2. Packet enlargement removes the pure-zero degeneracy

The new transfer board is no longer all-zero. On the `4 / 4` target packet, both baseline and defended threshold-transfer reads become:

- `AUC = 0.5`
- `ASR = 0.375`
- `TPR@1%FPR = 0.5`
- `TPR@0.1%FPR = 0.5`

So the earlier `1 / 1` all-zero read was at least partly a packet-size artifact.

### 3. But the defended checkpoint still shows zero improvement

Even after the minimal enlargement, the baseline and defended boards are numerically identical:

- `metric_deltas.auc = 0.0`
- `metric_deltas.asr = 0.0`
- `metric_deltas.tpr_at_1pct_fpr = 0.0`
- `metric_deltas.tpr_at_0_1pct_fpr = 0.0`

This means `H2` still does not expose any actual defense effect on the current same-contract transfer-only surface.

## Verdict

`negative but useful`

`04-H2` now has one honest minimal packet-scale follow-up. That follow-up proves the lane is not merely blocked by `1 / 1` degeneracy, but it still shows zero defended-vs-baseline separation.

So:

- `H2` remains below execution-ready successor status
- `H2` remains below `next_gpu_candidate`
- another same-contract packet-scale rerun is not justified right now

The highest-value immediate move is now a bounded stale-entry sync, because the active entry docs still point readers to `X-138` as if the follow-up were pending.

## Canonical Evidence Anchor

- `D:\Code\DiffAudit\Research\workspaces\implementation\runs\h2-review-defense-pilot-4x4-20260421-r1\summary.json`

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `current task closed = X-139 04-H2 4x4 bounded follow-up review`
- `next live lane = X-140 cross-box / system-consumable stale-entry sync after X-139`
- `CPU sidecar = I-A higher-layer boundary maintenance`

## Handoff

- `Research/ROADMAP.md`: yes
- `Research/README.md`: yes
- `docs/comprehensive-progress.md`: yes
- `docs/reproduction-status.md`: yes
- `docs/mainline-narrative.md`: yes
- `docs/research-autonomous-execution-prompt.md`: yes
- `workspaces/implementation/challenger-queue.md`: yes
- `Platform/Runtime`: no

Reason:

The result changes research-side control truth only.
