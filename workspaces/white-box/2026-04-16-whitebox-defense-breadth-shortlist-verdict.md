# 2026-04-16 White-Box Defense Breadth Shortlist Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `WB-3.1 / survey alternatives beyond DPDM`
- `current_admitted_defense`: `W-1 = DPDM strong-v3 full-scale`
- `decision`: `negative but useful`

## Question

Beyond the current `DPDM / W-1` defended comparator, is there already another materially different white-box defense candidate in the repo that is both distinct and executable on the admitted `DDPM/CIFAR-10` asset line?

## Reviewed Candidates

### `Finding NeMo`

Current repo-fixed status:

- `paper-faithful NeMo on current admitted white-box assets = no-go`
- `adapter-complete zero-GPU hold`
- `queue not-requestable`

Why it does not qualify:

1. it is a memorization-localization / observability route, not a released defended comparator;
2. its original protocol is `Stable Diffusion v1.4 / cross-attention value layers`, not the admitted `DDPM/CIFAR-10` line;
3. current repo evidence only authorizes bounded CPU observability exports, not a white-box defense benchmark.

### `Local Mirror`

Current repo-fixed status:

- already reviewed as collapsing back into the admitted `GSA` family.

Why it does not qualify:

1. it is not a defense candidate;
2. in the current repo context it is a literature alias / publication variant of the same gradient-based white-box attack family;
3. it therefore adds neither defense breadth nor a second defended family.

### `DPDM variants beyond admitted W-1`

Current status:

- `strong-v2`, `strong-v3`, same-protocol diagnostics, and shadow-count variants all exist in the repo.

Why they do not qualify as a second defense family:

1. they are still refinements inside `DPDM / W-1`, not a new mechanism;
2. they improve comparator depth and scale, but not defense diversity;
3. widening within `DPDM` is useful optimization, not breadth.

## Verdict

Current verdict:

- `negative but useful`

Reason:

1. the current repo does not already contain a second distinct executable white-box defense family;
2. `Finding NeMo` is still an observability / mechanism extension, not a released defended comparator;
3. `Local Mirror` does not add defense breadth at all;
4. the only currently executable white-box defense family remains `DPDM / W-1`.

## Decision

Current decision:

- `keep WB-3 open`
- `do not pretend a second defense family already exists`
- `treat the next white-box breadth step as candidate generation or import, not as immediate execution`

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: no direct change needed.
- Materials: if white-box defense breadth is mentioned, the honest wording is that the repo currently has one real defended family (`DPDM / W-1`) plus observability-side and literature-side candidates, but no second distinct executable defense family yet.
