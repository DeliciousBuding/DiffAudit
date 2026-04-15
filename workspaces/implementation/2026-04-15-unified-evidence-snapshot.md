# 2026-04-15 Unified Evidence Snapshot

## Scope

This snapshot consolidates the strongest current evidence across black-box, gray-box, white-box, and defense lines for competition use.

## Primary Evidence

| Line | Method | Dataset | Access | Best Current Rung | AUC | ASR | Status | Boundary |
|------|--------|---------|--------|-------------------|-----|-----|--------|----------|
| Black-box | Recon DDIM public-100 step30 | CelebA | generation-only | admitted runtime-mainline | `0.849` | `0.51` | admitted | controlled public-subset runtime evidence |
| Black-box best-single-metric | Recon DDIM public-50 step10 | CelebA | generation-only | narrower public-subset rung | `0.866` | `0.51` | candidate | satisfies the `>= 0.85` black-box strength gate |
| Black-box corroboration | CLiD clip local target100 | CelebA | generation-only + CLIP | local target-family rung | `1.0` | `1.0` | candidate | workspace-verified local bridge; target-side only |
| Black-box corroboration | CLiD clip local partial-target100 | CelebA | generation-only + CLIP | local target-family cross-check | `1.0` | `1.0` | candidate | workspace-verified local bridge; target-side only |
| Gray-box | PIA GPU512 adaptive baseline | CIFAR-10 | weights + gradients | runtime-mainline | `0.841339` | `0.786133` | runtime-mainline | workspace-verified; checkpoint provenance still blocks paper-faithful claim |
| Gray-box scale-up | PIA GPU1024 adaptive baseline | CIFAR-10 | weights + gradients | local scale-up rung | `0.83863` | `0.782715` | candidate | same protocol, larger local split cap |
| Gray-box corroboration | SecMI full split stat | CIFAR-10 | weights | full local split | `0.885833` | `0.815400` | candidate | full local split execution |
| Gray-box corroboration | SecMI full split NNS | CIFAR-10 | weights | full local split auxiliary head | `0.946286` | `0.879275` | candidate | strongest raw gray-box scorer in current local stack |
| White-box | GSA 1k-3shadow | CIFAR-10 | full weights + gradients | runtime-mainline | `0.998192` | `0.9895` | runtime-mainline | upper-bound style white-box evidence |

## Defense Snapshot

| Attack | Baseline | Defense | Defended AUC | Main Interpretation |
|--------|----------|---------|--------------|---------------------|
| PIA GPU512 | `0.841339` | stochastic dropout all_steps | `0.828075` | mild degradation, no collapse |
| PIA GPU1024 | `0.83863` | stochastic dropout all_steps | `0.825966` | same pattern at larger scale |
| GSA | `0.998192` | DPDM W-1 strong-v3 | `0.488783` | strong white-box defense comparator |

## Recommended Competition Framing

- Black-box mainline: keep `Recon public-100 step30` as the admitted headline, while citing `Recon public-50 step10` as the current best single-metric rung (`AUC 0.866`).
- Black-box corroboration: present `CLiD` as an independent local corroboration line that now survives two target-family checkpoints.
- Gray-box mainline: present `PIA` as the more controlled local baseline/defense ladder.
- Gray-box corroboration: present `SecMI` as the stronger alternate scorer proving the signal is not tied to one objective.
- White-box: present `GSA` as the upper-bound line and pair it with the strongest defended comparator.
