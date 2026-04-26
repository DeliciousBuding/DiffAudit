# 2026-04-15 Attack-Defense Matrix

## Matrix

| Attack Line | None / Baseline | Dropout | DPDM / dedicated defense | Current Read |
|-------------|------------------|---------|--------------------------|--------------|
| Recon | `0.849` | not yet re-run | not available | black-box headline remains strong |
| CLiD clip | `1.0` on local target-family rungs | not evaluated | not available | corroboration line is strong but still target-side |
| PIA GPU512 | `0.841339` | `0.828075` | not yet run | dropout weakens but does not neutralize |
| PIA GPU1024 | `0.83863` | `0.825966` | not yet run | same pattern under scale-up |
| TMIA-DM late-window | `0.839554` | `0.825867` | `0.733322` (`temporal-striding stride=2`) | strongest current defended gray-box challenger now prefers temporal-striding over dropout |
| SecMI stat | `0.885833` | not yet run | not yet run | full-split corroboration strong |
| SecMI NNS | `0.946286` | not yet run | not yet run | strongest current gray-box scorer |
| GSA | `0.998192` | not applicable | `0.488783` (`strong-v3`) | only line with a genuinely large defended drop |

## What The Matrix Says

- Black-box and gray-box both already show multi-line attack evidence.
- Gray-box dropout is consistent across scale but not transformative.
- The strongest defended gray-box challenger is no longer `TMIA + dropout`; it is now `TMIA + temporal-striding(stride=2)`.
- White-box currently has the strongest “attack vs defense” contrast.

## Coverage Note

- `not yet run` in this matrix means there is no current like-for-like defended comparator admitted into the final presentation package for that method, not that the research line is unfinished.
- Current defended headline evidence is centered on `PIA + stochastic dropout (all_steps)` for gray-box and `DPDM W-1` as the defended white-box comparator.
- Current defended gray-box challenger evidence should prefer `TMIA + temporal-striding(stride=2)` over the older `TMIA + dropout` comparison.
- `Recon` and `CLiD` are currently used as black-box leakage evidence, not as a black-box defense benchmark ladder.
- `CLiD` should now be described as `evaluator-near local clip-only corroboration`, not as a paper-faithful black-box threshold evaluator.
- `variation` should now be described as `contract-ready blocked`, not generic blocked-assets.
- `SecMI` is currently used as a strong gray-box corroboration scorer; there is no matched defended `SecMI` rung in the final package.
- Reopened negative result:
  - `epsilon-precision-throttling` on `PIA cpu-32` was tested as a materially different gray-box defense candidate on `2026-04-15`, but it increased `AUC` / `ASR` versus the paired baseline and is currently rejected as a near-term defended mainline candidate.
  - `served-image-sanitization` on local `CLiD` (`JPEG 70`, `512 -> 448 -> 512`, `32 / 32` bounded probe) was tested on `2026-04-15`, but it preserved `AUC / ASR / TPR@1%FPR = 1.0 / 1.0 / 1.0` and is currently rejected as a near-term black-box mitigation mainline candidate.
- White-box defense breadth remains single-family:
  - there is still no second executable defended white-box family beyond `W-1 = DPDM`

## Next Best Defense Work

1. Add a materially different gray-box defense instead of another dropout variant.
2. Only revisit black-box defenses if we have a materially different query-side or serving-side mitigation concept beyond mild JPEG-style sanitization.
