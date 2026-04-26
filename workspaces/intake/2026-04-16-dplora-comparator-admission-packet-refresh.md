# 2026-04-16 DP-LoRA Comparator Admission Packet Refresh

## Question

After `WB-12` refreshed the release-review logic around the completed same-asset local comparator board, how should the stale `baseline vs SMP-LoRA vs W-1` admission packet itself be rewritten?

## Inputs

- `workspaces/white-box/2026-04-16-dplora-comparator-release-review-refresh.md`
- `workspaces/white-box/2026-04-16-dplora-local-board-refresh-verdict.md`
- `workspaces/intake/2026-04-14-baseline-smp-lora-w1-comparator-admission-packet.md`

## Stale Fields Detected

The old packet is no longer the best truthful object because it still centers:

- `batch14 throughput` rescue framing
- pre-local-board stop conditions
- pre-refresh rung wording

That packet was honest when the board had not yet been executed.

It is now stale because the repo already has:

- one completed same-asset local comparator board
- one bounded `W-1 local63` refresh
- one stronger release-review reading

## What The Packet Must Now Say

The refreshed packet should freeze:

1. comparator board
   - `baseline local63`
   - frozen `SMP-LoRA local63`
   - `W-1 strong-v3 local63 refresh`
2. current local ordering on the shared primary metric
   - `SMP-LoRA (0.34375) < W-1 local63 (0.474175) < baseline (0.5565217391304348)`
3. boundary
   - this is a bounded local comparator win
   - not an admitted white-box upgrade
   - not a full-scale benchmark replacement
4. release gate
   - no automatic new GPU release follows from this packet

## Practical Consequence

This packet should stop asking:

- whether a local comparator board exists
- whether the old `batch14 throughput` rescue sweep should be the main comparator object

It should start asking:

- whether any further question remains after the bounded local-board win
- and if so, whether that next question is CPU-side metric harmonization or a genuinely new bounded GPU hypothesis

## Verdict

- `admission_packet_refresh = positive`
- `old_packet_status = stale`
- `new_packet_should_center = completed-local-comparator-board`
- `gpu_release = none`
- `next_step = rewrite the comparator admission packet to match current local-board truth`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this updates intake truth and release discipline only.
