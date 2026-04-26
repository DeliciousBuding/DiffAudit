# 2026-04-16 Gray-Box New-Family Selection: SimA

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `GB-3.1 / GB-3.2`
- `selected_family`: `SimA`
- `gpu_status`: `not requested`
- `verdict`: `selected for next feasibility implementation`

## Question

After the gray-box second-defense shortlist stalled on repeated negative perturbation candidates, which new gray-box family should become the next active branch?

## Candidates Considered

Primary literature-side candidates already named in the repo:

1. `SimA`
2. `TMIA-DM`
3. `MoFit`

## Selection Verdict

Selected next family:

- `SimA`

## Why SimA Wins

1. it is materially different from `PIA`:
   - `PIA` uses multi-step `epsilon-trajectory consistency`
   - `SimA` uses a single-query score / noise-norm statistic
2. it already fits the repo's gray-box signal-axis narrative, so the method story stays coherent instead of introducing a new threat-model digression;
3. it appears execution-feasible on the current local CIFAR-10 DDPM assets because the repo already has:
   - canonical denoiser access through the current `PIA` runtime path
   - member / non-member splits
   - ROC / threshold evaluation plumbing
4. compared with `TMIA-DM`, it has a shorter implementation path and fewer new moving parts;
5. compared with `MoFit`, it does not require jumping into text-conditioned latent diffusion assets or caption/embedding machinery.

## Why The Others Lose For Now

### `TMIA-DM`

- still attractive as a time-dependent signal family;
- but current repo state lacks an equally short local execution path.

### `MoFit`

- attractive for long-horizon latent-diffusion expansion;
- but it is much farther from the current CIFAR-10 DDPM mainline and would force a larger asset/protocol jump.

## Shortest Repo-Grounded Path

Shortest bounded implementation path for `SimA`:

1. reuse the current DDPM denoiser-loading path already exercised by `pia_adapter.py`;
2. expose a single-query score-norm computation at chosen timestep `t`;
3. start with a CPU bounded rung and a local timestep scan rather than a large GPU run;
4. reuse the current summary/ROC-style output contract so the new family can be compared honestly against `PIA` and `SecMI`.

## First Feasibility Step

Recommended first bounded step:

- implement a `cpu-32` or similarly bounded `SimA` probe on the current CIFAR-10 DDPM asset line with timestep scan over a small early-to-mid window.

Expected output:

- one feasibility note
- one run `summary.json`
- one verdict on whether `SimA` is locally strong enough to join gray-box challenger status

## Handoff Note

- `Platform`: no direct change needed yet.
- `Runtime`: no direct change needed yet.
- Materials: this selection lets gray-box wording pivot from “we tried more defense knobs” to “we are now adding a genuinely different gray-box signal family.”
