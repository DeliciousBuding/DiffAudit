# 2026-04-16 TMIA-DM Protocol And Asset Note

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `TMIA-DM follow-up decomposition`
- `track`: `gray-box`
- `decision_grade`: `decision-grade`
- `current_verdict`: `protocol-ready but not execution-released`
- `gpu_release`: `none`

## Question

After reselection, what exactly must be true for `TMIA-DM` to move from literature-side candidate into a bounded local implementation question?

## Current Verdict

Current verdict:

- `protocol-ready but not execution-released`

Interpretation:

1. `TMIA-DM` is no longer only “paper archived / threat-model judged”;
2. the repo now has enough information to define:
   - the signal surface
   - the access assumption
   - the local-fit path
   - the minimal smoke entry
3. but it still does **not** have a released execution path or GPU justification.

## Signal Surface

The paper's stable method-level claim is:

1. short-duration attacks benefit more from `noise gradient` information;
2. long-duration attacks benefit more from `temporal noise` information;
3. the final method is a threshold-style gray-box classifier over those time-dependent features.

For local repo purposes, the safest decomposition is:

- `short-window feature family`
  - adjacent or near-adjacent timestep change features over predicted `epsilon`
- `long-window feature family`
  - multi-timestep temporal aggregation features over predicted `epsilon`

This means the first local implementation should treat `TMIA-DM` as:

- a `multi-timestep gray-box feature extractor`
- not a one-step score like `SimA`
- not a replay-consistency attack exactly like `PIA`

## Access Assumption

Relative to current local lines:

- stronger than strict black-box `recon / variation`
- comparable in privilege to `PIA / SimA`
- likely lighter than a full white-box gradient route

Minimum access the repo would need:

1. direct denoiser query at chosen timestep `t`
2. ability to query multiple timesteps for the same sample
3. stable member / non-member split for threshold evaluation

This keeps `TMIA-DM` firmly inside gray-box, not black-box.

## Local Fit

Closest current local seam:

- `Research/src/diffaudit/attacks/pia_adapter.py`

Why:

1. it already loads the same DDPM/CIFAR-10 canonical checkpoint line;
2. it already exposes denoiser access through the current `PIA` runtime path;
3. it already owns the local split loading, score dumping, and threshold metric contract.

So the shortest local path is **not**:

- `recon`
- `variation`
- `CLiD`

It is:

- a new gray-box adapter parallel to `PIA` and `SimA`, but reusing the same canonical assets.

## Asset Checklist

Current asset read:

1. target model:
   - current `PIA` canonical DDPM checkpoint root is sufficient for a first bounded probe
2. member / non-member split:
   - current `PIA` canonical split is sufficient
3. timestep access:
   - current local DDPM path already supports repeated timestep queries through the loaded denoiser
4. missing asset:
   - none decisive for a bounded CPU protocol probe

So the remaining blocker is no longer “asset root unknown”.
It is “feature definition and implementation path not yet fixed”.

## Minimal Smoke Entry Definition

The first honest local smoke should be defined as:

- `mode = cpu bounded protocol probe`
- `dataset = current CIFAR-10 DDPM canonical asset line`
- `sample_count_per_split = 32`
- `feature groups`:
  - one short-window temporal-difference feature family
  - one long-window temporal-aggregation feature family
- `output`:
  - per-family metrics
  - fused metric if and only if both feature groups are implemented honestly

The key boundary is:

- this first smoke is allowed to use a repo-grounded proxy for `noise gradient` if the exact paper feature is still ambiguous,
- but it must be labeled clearly as:
  - `protocol-probe`
  - not `paper-faithful reproduction`

## Execution Release Rule

Current release rule:

- `execution_release = none`
- `gpu_release = none`

Reopen to implementation only when one concrete next-step note names:

1. exact local feature definitions
2. exact timestep windows
3. exact summary schema
4. what would count as:
   - `positive`
   - `negative`
   - `not-yet`

## Recommended Next Step

Recommended next bounded task:

- implement a `TMIA-DM protocol probe` on CPU that compares:
  - short-window temporal-difference features
  - long-window temporal-aggregation features

Expected outcome:

- either a credible local feature path worth keeping,
- or a stronger `not-yet / no-go` verdict that closes the branch honestly.

## Handoff Note

- `Platform`: no immediate change needed.
- `Runtime`: no immediate change needed.
- Materials: wording can now say `TMIA-DM` has moved beyond generic literature support into a protocol-ready gray-box candidate, while still not being execution-released.
