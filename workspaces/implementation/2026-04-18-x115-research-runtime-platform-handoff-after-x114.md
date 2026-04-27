# 2026-04-18 X-115 Research Runtime Platform Handoff After X-114

## Question

After `X-114` sharpened `04-defense` to `CPU-first family review`, does `Research -> Runtime -> Platform` now require any new field, endpoint, snapshot contract, or runner capability?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-18-x114-04-defense-post-h1-family-review.md`
- `<DIFFAUDIT_ROOT>/Research/docs/local-api.md`
- `<DIFFAUDIT_ROOT>/Research/docs/reproduction-status.md`
- `<DIFFAUDIT_ROOT>/Platform/RUNBOOK.md`
- `<DIFFAUDIT_ROOT>/Platform/docs/public-runtime-handoff.md`
- `<DIFFAUDIT_ROOT>/Runtime-Server/README.md`
- `<DIFFAUDIT_ROOT>/Platform/apps/api-go/README.md`
- `<DIFFAUDIT_ROOT>/Platform/apps/api-go/scripts/publish_public_snapshot.py`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-18-platform-intake-from-research.md`

## What Was Checked

### 1. Current admitted read path

The current system-consumable read path is already explicit:

1. admitted cross-track surface = `Research/workspaces/implementation/artifacts/unified-attack-defense-table.json`
2. Runtime endpoint = `GET /api/v1/evidence/attack-defense-table`
3. Platform public snapshot = `apps/api-go/data/public/attack-defense-table.json`
4. publish-time fallback = direct sync from the same Research admitted table when Runtime is unavailable

### 2. What `X-114` actually changed

`X-114` changed:

1. `04-defense` control-plane reading
2. higher-layer wording about `H1` vs `H2`
3. the honest next-step posture for future Research sessions

`X-114` did **not** change:

1. any admitted metric row
2. any admitted table schema
3. any Runtime API route
4. any Platform snapshot shape
5. any runner capability requirement

## Actual Read

### 1. Runtime does not need a new contract

Current Runtime responsibility is still:

1. expose admitted evidence
2. keep research-root-backed read endpoints stable
3. optionally support audit control-plane execution

The `04` update is not yet a runner-facing promotion.

There is still:

1. no `04-H2` executable contract
2. no new runner job type
3. no new endpoint request shape

So Runtime should **not** add a new `04` contract surface at this stage.

### 2. Platform does not need a new display field

Current Platform already has the right admitted display surface:

1. `catalog.json`
2. `attack-defense-table.json`
3. existing boundary / evidence / provenance rendering

`X-114` only sharpens planning truth:

1. `04` is still a research-side lane
2. `H2` is still fallback wording only
3. there is still no new admitted result to show publicly

So Platform should not create:

1. a new `04-H2` badge
2. a new planning-derived table row
3. a new consumer field for `family_review_state`

### 3. The right handoff is wording-level, not protocol-level

The honest handoff is:

1. future sessions should not misread `H2` as execution-ready
2. Runtime / Platform should continue reading admitted data only
3. if `H2` later becomes executable, that later step should reuse the existing admitted/candidate governance pattern before asking for new consumer fields

## Verdict

- `x115_research_runtime_platform_handoff_after_x114_verdict = contract-stable wording-sync only`

More precise reading:

1. `X-114` changes control-plane truth, not consumer schema
2. Runtime needs no new endpoint, field, or runner capability
3. Platform needs no new snapshot shape or UI field
4. the correct action is to keep the current admitted read path and carry the sharper `04` boundary only in research-side notes/prompting

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `Runtime handoff = none beyond note-level alignment`
- `Platform handoff = none beyond note-level alignment`
- `competition-material sync decision = none`

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-18-x115-research-runtime-platform-handoff-after-x114.md`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Research/docs/research-autonomous-execution-prompt.md`: update required
- `Platform/Runtime code change`: not required
