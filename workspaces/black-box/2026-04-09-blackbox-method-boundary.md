# 2026-04-09 Black-Box Note: Method Boundary After TMIA-DM Intake

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-09 +08:00`
- `current_state`: `black-box hierarchy rechecked after TMIA-DM intake`

## A. Frozen Black-Box Hierarchy

Main evidence:

- `recon DDIM public-100 step30`

Best single-metric reference:

- `recon DDIM public-50 step10`

Formal local secondary track:

- `variation / Towards`
- status: `formal local secondary track + blocked real-API assets`

Supplementary black-box-adjacent material:

- `CLiD`
- face-image LDM black-box paper

## B. What TMIA-DM Changes

It changes the literature picture, not the black-box execution hierarchy.

After intake, `TMIA-DM` should be read as:

- a `research-ready` gray-box candidate
- another paper arguing that member signal can live in time-dependent noise behavior
- closer to `PIA` / temporal-noise gray-box logic than to API-only black-box logic

So the correct boundary is:

- `TMIA-DM` strengthens the repo's understanding of diffusion-time membership signals
- `TMIA-DM` does **not** become a black-box mainline, secondary track, or runnable black-box candidate

## C. Why It Is Not Strict Black-Box

The paper explicitly describes:

- `query-based gray-box attack`
- direct use of temporal noise and gradient information
- access to intermediate diffusion-time information rather than only final image outputs

That is stronger access than:

- `recon`
- `variation / Towards`

So even if it is query-driven, it is still not a strict black-box line in this repository's taxonomy.

## D. Allowed External Wording

Allowed:

- black-box currently remains `recon + variation`
- `TMIA-DM` is a newly archived gray-box candidate paper
- it supports the same broad temporal/noise signal direction already visible in `PIA`

Not allowed:

- `TMIA-DM` is our new black-box method
- `TMIA-DM` replaces `variation`
- `TMIA-DM` proves black-box can use temporal noise in the current repo

## E. Next Step

1. Keep `recon` and `variation` wording unchanged in black-box status pages.
2. Use `TMIA-DM` only to strengthen gray-box literature discussion.
3. Continue black-box work on `recon` semantics and `variation` real-asset gating.
