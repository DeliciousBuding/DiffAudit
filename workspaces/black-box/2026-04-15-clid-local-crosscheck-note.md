# 2026-04-15 Black-Box Note: CLiD Local Cross-Check Across Recon Target Families

## Scope

This note consolidates the local `CLiD clip` bridge runs completed on the two available Recon target-family checkpoints.

## Local Rungs

| Run | Checkpoint | Split | AUC | ASR | Notes |
|-----|------------|-------|-----|-----|------|
| `clid-recon-clip-target100-20260415-r1` | `celeba_target/checkpoint-25000` | `100 / 100` | `1.0` | `1.0` | First scaled local rung after bridge smoke |
| `clid-recon-clip-partial-target100-20260415-r1` | `celeba_partial_target/checkpoint-25000` | `100 / 100` | `1.0` | `1.0` | Cross-check rung on the partial-target checkpoint family |

## Main Takeaways

- The current local `CLiD clip` signal is not restricted to a single Recon target checkpoint.
- Both available target-family LoRA checkpoints produce the same clean `100 / 100` separation under the local bridge.
- This is enough to move the question from “can CLiD run here?” to “how should we present and validate this line?”

## Boundary

- These are still target-side local bridge rungs, not paper-faithful CLiD replications.
- No shadow-member rung is available from the current local Recon asset bundle.
- Competition wording should keep the boundary at `workspace-verified local black-box corroboration`.

## Recommended Next Step

- Prefer `Recon + CLiD` black-box comparison or late fusion over spending more time on another target-side CLiD rung.
