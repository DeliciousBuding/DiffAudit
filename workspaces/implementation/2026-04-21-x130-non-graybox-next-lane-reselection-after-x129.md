# 2026-04-21 X-130 Non-Graybox Next-Lane Reselection After X-129

## Question

After `X-129` cleared the remaining material-facing `I-A` wording softness, should the next live lane move straight into a fresh candidate-surface expansion, or is there still one active system-consumable stale-entry layer that must be cleared first?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/README.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`
- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-21-x129-ia-truth-hardening-after-x128.md`

## Findings

### 1. One active system-consumable drift still remains

Two still-active entry surfaces are stale:

- `README.md`
  - still says the next gray-box CPU-first lane is `PIA vs TMIA-DM confidence-gated switching design review`
- `workspaces/implementation/challenger-queue.md`
  - still marks `WB-CH-4` as `actual-packet-pending`
  - still carries a stale `Recommended Next Order` headed by `X-86`

These are no longer archival-only wording artifacts; they are live read-path surfaces that can mis-steer fresh sessions.

### 2. Therefore the next move is not a fresh candidate expansion yet

Because those stale surfaces are still active and operator-facing, the highest-value immediate move is:

- one bounded `cross-box / system-consumable stale-entry sync`

not:

- another empty reselection loop
- or a new candidate-surface expansion before the live read-path is corrected

## Verdict

`positive`.

Sharper control truth:

1. current non-graybox reselection still exposes one remaining active stale-entry layer
2. the next honest live lane becomes:
   - `X-131 cross-box / system-consumable stale-entry sync after X-130 reselection`
3. `active_gpu_question = none`
4. `next_gpu_candidate = none`

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-21-x130-non-graybox-next-lane-reselection-after-x129.md`

## Handoff

- `Research/ROADMAP.md`: yes
- `workspaces/implementation/challenger-queue.md`: yes
- `README.md`: yes
- `Platform/Runtime`: no

Reason:

This is a research-side lane-ordering and stale-entry decision only. It does not alter admitted metrics or runtime contracts.
