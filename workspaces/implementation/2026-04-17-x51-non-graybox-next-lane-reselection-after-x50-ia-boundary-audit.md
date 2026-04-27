# 2026-04-17 X-51 Non-Graybox Next-Lane Reselection After X-50 I-A Boundary Audit

## Question

Once `X-50` clears the remaining `I-A` higher-layer drift, what is the most honest next non-graybox lane:

- continue splitting `I-A`
- reopen a blocked/hold branch
- or clear one more stale system/material-facing surface first

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`
- `<DIFFAUDIT_ROOT>/Research/docs/mainline-narrative.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/docs/competition-evidence-pack.md`

## Reselection Read

After `X-50`, no stronger blocked/hold non-graybox branch honestly reopened:

- `XB-CH-2` still remains `needs-assets`
- `GB-CH-2` still remains `hold`
- white-box still does not expose a new executable successor
- `I-A` no longer has an unresolved packet-truth gap; the remaining visible issue is one materials-facing stale-entry surface

That remaining stale-entry surface is concrete:

- `competition-evidence-pack.md` still encodes `SecMI` as `blocked baseline`
- the same file still encodes `TMIA-DM` as `intake only`

Both are below current repo truth and can mislead leader/material consumers even if the internal research docs are already aligned.

## Verdict

- `x51_non_graybox_next_lane_reselection_verdict = positive`

Selected next lane:

- `X-52 cross-box / materials stale-entry sync after X-51 reselection`

## Control-Plane Result

- `active GPU question = none`
- `next_gpu_candidate = none`
- `current CPU sidecar = I-A higher-layer boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Research/workspaces/implementation/challenger-queue.md`: update required
- `Research/docs/competition-evidence-pack.md`: update required by the selected next lane
- `Leader/materials`: sync suggested, because this is a materials-facing stale-entry issue
