# 2026-04-16 Gray-Box Next-Family Reselection

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `gray-box follow-up after SimA`
- `selected_next_branch`: `TMIA-DM protocol / asset decomposition`
- `gpu_status`: `none`

## Question

After the bounded `SimA` feasibility run closed as `negative but useful`, which gray-box branch should become the next active task?

## Candidates Reviewed

1. `SimA reopen`
2. `TMIA-DM`
3. `MoFit`

## Decision

Selected next branch:

- `TMIA-DM protocol / asset decomposition`

## Why This Branch Wins

1. `SimA reopen` is still possible, but only with a fresh paper-faithful hypothesis; reopening immediately would likely just rescan the same weak local path.
2. `MoFit` is methodologically interesting, but it jumps to text-conditioned latent diffusion and caption/embedding machinery, which is too large a protocol jump for the next bounded step.
3. `TMIA-DM` stays on the current gray-box time/noise signal axis and is closer to the existing DDPM/CIFAR-10 mainline.
4. The repo already has an intake decomposition note for `TMIA-DM`, so the next action can be a bounded non-GPU upgrade rather than a vague literature revisit.

## Immediate Next Step

Recommended next task:

- write the next `TMIA-DM` protocol note that turns the current `not-yet` intake into a concrete:
  - signal surface
  - access assumption
  - local-fit mapping
  - minimal smoke entry definition

Expected outcome:

- either a credible bounded implementation path
- or a stronger `not-yet / no-go` verdict

## Handoff Note

- `Platform`: no change needed.
- `Runtime`: no change needed.
- Materials: gray-box wording can now honestly say the branch strategy after `SimA` is to move deeper into a time/noise family with better local fit, rather than force latent-diffusion machinery too early.
