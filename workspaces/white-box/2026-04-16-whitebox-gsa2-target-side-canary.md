# 2026-04-16 White-Box GSA2 Target-Side Canary

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `WB-2.1 / WB-2.2`
- `selected_path`: `GSA2 comparator`
- `track`: `white-box`
- `gpu_status`: `completed`

## Question

After `WB-1` removed the vague “gradient extraction may be broken” blocker, which second white-box branch should stay alive first?

## Decision

Choose:

- `GSA2 comparator`

Do not reopen first:

- `Finding NeMo`
- `Local Mirror`

Reason:

1. `Finding NeMo` is already repo-fixed to `blocked / no-go for current execution wave`;
2. `Local Mirror` still collapses back into the admitted `GSA` family;
3. `GSA2` is the only branch that now has a short, bounded, repo-grounded execution path after `WB-1`.

## Canary Runs

Target-member direct extraction:

- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-gradient-direct-attackmethod2-20260415-r2\target_member-gradients.pt`

Target-nonmember direct extraction:

- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-gradient-direct-attackmethod2-20260415-r3\target_nonmember-gradients.pt`

Shared canary contract:

- upstream entrypoint:
  - `workspaces/white-box/external/GSA/DDPM/gen_l2_gradients_DDPM.py`
- `attack_method = 2`
- admitted target checkpoint root
- admitted target-member / target-nonmember imagefolder assets

## Verdict

Current verdict:

- `positive canary`

Interpretation:

1. `GSA2` remains execution-eligible on the current admitted target-side assets;
2. the repo should now treat `GSA2 comparator` as the live `WB-2` branch;
3. the remaining work is no longer “can gradients be extracted at all”, but “is the full comparator worth promotion?”

## Boundary

This is **not** yet a second white-box benchmark verdict.

What is proven:

- target-side `attack_method = 2` extraction works for both member and non-member target splits.

What is not yet proven:

- full shadow-backed comparator value,
- distinct project-level value versus admitted `GSA1`,
- promotion into a formal second white-box mainline.

## Next Best Move

1. use the same `GSA2` direct extraction contract for shadow-side canaries or a bounded comparator;
2. stop spending time on “is GSA2 fundamentally broken?”;
3. spend the next white-box budget on comparator value, not on re-proving extraction viability.
