# 2026-04-16 DP-LoRA Comparator Contract Reconciliation

## Question

After `WB-6`, is the currently written `baseline vs SMP-LoRA vs W-1` comparator packet still aligned with the now-frozen minimal local `DP-LoRA` candidate, or does the comparator contract need reconciliation before any future release review can be honest?

## Inputs

- `workspaces/white-box/2026-04-16-dplora-minimal-local-config-candidate.md`
- `workspaces/white-box/2026-04-16-dplora-no-go-and-gpu-release-triggers.md`
- `workspaces/white-box/2026-04-16-dplora-comparator-release-review.md`
- `workspaces/intake/2026-04-14-baseline-smp-lora-w1-comparator-admission-packet.md`
- `outputs/smp-lora-sweep/sweep_results.json`
- `outputs/smp-lora-phase2/baseline_nodefense_target-64/evaluation.json`

## Mismatch Found

The current comparator packet is directionally right but contract-stale.

Why:

1. `WB-5` froze the minimal local candidate to:
   - `lambda=0.1 / rank=4 / epochs=10`
2. the current comparator packet still centers its execution examples on:
   - `T06 batch14 throughput AdamW / SGD`
3. those `T06` outputs belong to a later optimization frontier and are no longer the cleanest frozen local candidate contract

This means the current packet is not yet the best possible release-review object.

## Reconciled Comparator Board

The comparator board should now be read as:

- `baseline`:
  - local no-defense evaluation
  - current anchor `AUC = 0.5565217391304348`
- `SMP-LoRA local candidate`:
  - frozen candidate `lambda=0.1 / rank=4 / epochs=10`
  - current local anchor `AUC = 0.34375`
- `W-1 reference`:
  - `DPDM strong-v3 full-scale`

The historical `T06 batch14 throughput` frontier remains useful only as:

- supporting context for why optimizer/lr rescue is already closed
- not as the primary comparator candidate itself

## Practical Consequence

Before any future GPU release review, the comparator packet should be interpreted and refreshed around:

- `baseline vs frozen SMP-LoRA local candidate vs W-1`

not around:

- `optimizer/lr rescue remnants`

This is still a CPU-side contract action, not a GPU release.

## Verdict

- `contract_reconciliation_verdict = positive`
- `stale_packet_detected = yes`
- `reconciled_comparator_board = baseline-vs-frozen-smp-lora-vs-w1`
- `gpu_release = none`
- `next_step = use the reconciled board as the only honest future comparator release-review surface`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this prevents queue drift and wrong-GPU-release logic, but it does not change admitted white-box claims.
