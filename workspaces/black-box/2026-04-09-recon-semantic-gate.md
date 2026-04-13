# 2026-04-09 Black-Box Note: Recon Semantic Gate

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-09 +08:00`
- `selected_track`: `recon`
- `current_state`: `semantic gate clarified`

## A. Current Semantic Gate

The current local `recon` line is strong enough to support:

- `main evidence`
- `best single metric reference`

It is **not** strong enough to support:

- fully paper-aligned `target/shadow/member/non-member` semantics

The blocking point remains:

- `shadow_member` is still a proxy semantic mapping

## B. Current Most Defensible Mapping

As of now:

- `target_member`
  - `partial-100-target/member`
- `target_non_member`
  - `partial-100-target/non_member`
- `shadow_non_member`
  - `100-shadow/non_member`
- `shadow_member`
  - proxy only, currently approximated with `100-target/non_member`

This is enough for a local defended black-box evidence chain.
It is not enough for a full claim that the public asset bundle exactly matches the paper's four-way split semantics.

Additional support now exists in:

- `derived-public-10/mapping-note.md`
- `derived-public-25/mapping-note.md`
- `derived-public-50/mapping-note.md`
- `derived-public-100/mapping-note.md`

These files explicitly record the current local mapping as `shadow_member_proxy`, which strengthens local consistency but still does not remove the semantic gate.

## C. What Is Allowed To Say

Allowed:

- `recon DDIM public-100 step30` is the current most complete black-box main evidence
- `recon DDIM public-50 step10` is the best single-metric reference
- current public assets are sufficient for a local runtime-mainline evidence chain

Not allowed:

- the public bundle has fully confirmed `target/shadow/member/non-member` semantics
- `shadow_member` is already paper-validated
- `step30` universally dominates `step10`

## D. Promotion Gate

The semantic gate can only be promoted when at least one of these appears:

1. official split documentation from the asset source
2. a more complete public bundle with explicit `shadow_member`
3. upstream code or paper appendix that resolves the current four-way mapping ambiguity

Until then:

- keep `recon` as admitted main evidence
- keep the semantic claim explicitly constrained

## E. Immediate Next Step

1. Keep [recon-public-asset-mapping.md](../../docs/recon-public-asset-mapping.md) as the source of truth.
2. Use this note as the short-form gate whenever the black-box mainline is summarized.
3. Do not spend GPU on recon reruns until the semantic gate, not just the metric table, is the real bottleneck.
