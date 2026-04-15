# 2026-04-15 Attack-Defense Matrix

## Matrix

| Attack Line | None / Baseline | Dropout | DPDM / dedicated defense | Current Read |
|-------------|------------------|---------|--------------------------|--------------|
| Recon | `0.849` | not yet re-run | not available | black-box headline remains strong |
| CLiD clip | `1.0` on local target-family rungs | not evaluated | not available | corroboration line is strong but still target-side |
| PIA GPU512 | `0.841339` | `0.828075` | not yet run | dropout weakens but does not neutralize |
| PIA GPU1024 | `0.83863` | `0.825966` | not yet run | same pattern under scale-up |
| SecMI stat | `0.885833` | not yet run | not yet run | full-split corroboration strong |
| SecMI NNS | `0.946286` | not yet run | not yet run | strongest current gray-box scorer |
| GSA | `0.998192` | not applicable | `0.488783` (`strong-v3`) | only line with a genuinely large defended drop |

## What The Matrix Says

- Black-box and gray-box both already show multi-line attack evidence.
- Gray-box dropout is consistent across scale but not transformative.
- White-box currently has the strongest “attack vs defense” contrast.

## Coverage Note

- `not yet run` in this matrix means there is no current like-for-like defended comparator admitted into the final presentation package for that method, not that the research line is unfinished.
- Current defended headline evidence is centered on `PIA + stochastic dropout (all_steps)` for gray-box and `DPDM W-1` as the defended white-box comparator.
- `Recon` and `CLiD` are currently used as black-box leakage evidence, not as a black-box defense benchmark ladder.
- `SecMI` is currently used as a strong gray-box corroboration scorer; there is no matched defended `SecMI` rung in the final package.

## Next Best Defense Work

1. Add a materially different gray-box defense instead of another dropout variant.
2. Only revisit black-box defenses if we have a concrete query-side mitigation concept.
