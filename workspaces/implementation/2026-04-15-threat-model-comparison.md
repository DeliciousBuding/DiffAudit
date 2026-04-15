# 2026-04-15 Threat Model Comparison

## Table

| Attack | Model Access | Data Access | Extra Tooling | Current Best Local Evidence | Practical Meaning |
|--------|--------------|-------------|---------------|-----------------------------|-------------------|
| Recon | generation-only | target/shadow image sets | image encoder + reconstruction pipeline | `AUC 0.849` | strongest admitted black-box line |
| CLiD clip | generation-only + CLIP-style conditional scoring | text-image pairs | SD1.5 base snapshot + CLIP path | local `100/100` target-family corroboration | black-box corroboration on same asset family |
| PIA | weights + score/gradient-side access | known target split | repeated timestep probing | `AUC 0.841339` at 512, `0.83863` at 1024 | controlled local gray-box mainline |
| SecMI | weights | known split + feature extractor | ResNet features + statistical tests | `AUC 0.885833` stat, `0.946286` NNS | strongest current alternate gray-box scorer |
| GSA | full weights + gradients + shadow data | shadow/member splits | gradient extraction + classifier | `AUC 0.998192` | white-box upper-bound attack |

## Narrative Use

- Use this table to explain that higher attack strength generally tracks stronger access assumptions.
- Pair `Recon` with `CLiD` to show black-box evidence does not rest on one mechanism.
- Pair `PIA` with `SecMI` to show gray-box evidence does not rest on one objective.
- Use `GSA` to illustrate the upper bound once the adversary gets full white-box access.
