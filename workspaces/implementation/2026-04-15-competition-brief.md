# 2026-04-15 Competition Brief

## One-Sentence Story

DiffAudit now has converging evidence that diffusion-model membership risk is observable across black-box, gray-box, and white-box settings, and the strongest currently tested lightweight gray-box defense only weakens, not removes, that signal.

## What We Can Defend Live

### Black-box

- `Recon` remains the admitted headline line with `AUC 0.849`.
- The stricter “strong black-box result” gate is already met by `Recon public-50 step10` with `AUC 0.866`.
- `CLiD clip` is now locally runnable and corroborated across two target-family Recon checkpoints, each with `100 / 100` target-side separation.
- Honest boundary: current `CLiD` evidence is `workspace-verified local corroboration`, not a paper-faithful full CLiD benchmark.

### Gray-box

- `PIA` is the best controlled local runtime mainline.
- The signal is stable when scaling from `512 / 512` to `1024 / 1024` (`0.841339 -> 0.83863`).
- `SecMI` independently corroborates the gray-box risk and currently gives the strongest raw full-split score (`stat 0.885833`, `NNS 0.946286`).

### White-box

- `GSA` remains the upper-bound line with near-saturated attack strength (`AUC 0.998192`).
- This is the clearest proof that privileged access is catastrophic.

## Defense Bottom Line

- `PIA` + stochastic dropout (`all_steps`) shows a repeatable but limited reduction at both `512` and `1024` scale.
- The strongest current “attack vs defense” visual contrast is still on the white-box side (`GSA` vs `DPDM W-1`).
- Honest claim: our present lightweight gray-box defense mitigates but does not neutralize leakage.

## Recommended Slide Wording

1. Black-box: “Externally observable leakage is not tied to one attack mechanism; reconstruction and CLIP-style conditional discrepancy both light up on our local asset family.”
2. Gray-box: “The gray-box signal is stable under scale-up and independently corroborated by SecMI.”
3. White-box: “Once privileged access is available, membership signal becomes near-trivial.”
4. Defense: “Simple stochastic dropout helps, but stronger defenses are still needed.”

## Do Not Overclaim

- Do not call the local `CLiD` runs paper-faithful replications.
- Do not present `PIA` or `SecMI` as benchmark-ready beyond the documented workspace boundary.
- Do not present any current result as proof of generalized internet exploitability.
