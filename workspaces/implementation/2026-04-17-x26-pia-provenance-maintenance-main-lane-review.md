# 2026-04-17 X-26 PIA Provenance Maintenance Main-Lane Review

## Question

Now that `PIA provenance maintenance` has been promoted from sidecar to the main `CPU-first` lane, does the current provenance blocker still require new machine-readable consumer fields or a new execution release, or is the honest next step to freeze the carry-forward boundary and reopen triggers?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/pia-intake-gate.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/assets/pia/PROVENANCE.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-09-pia-provenance-dossier.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-10-pia-provenance-upstream-identity-note.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-10-pia-provenance-split-protocol-delta.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-09-pia-paper-alignment-gap.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/assets/pia/manifest.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/intake/index.json`
- `<DIFFAUDIT_ROOT>/Research/docs/admitted-results-summary.md`
- `<DIFFAUDIT_ROOT>/Research/docs/asset-registry-local-api.md`

## Review

### 1. The blocker tuple is already sharp enough

The current `PIA` provenance blocker is no longer vague.

It is already reducible to:

1. `release/source identity unresolved`
2. `CIFAR10 random-four-split / four-model tau-transfer protocol mismatch vs current local single fixed split`

And one hygiene caveat must remain explicit:

- the `2026-04-09` strict gate is a historical clean snapshot, not a present-tense release identity proof

### 2. The machine-readable intake surface is already sufficient

The current intake surface already freezes the right machine-readable facts:

- `contract_key = gray-box/pia/cifar10-ddpm`
- `asset_grade = single-machine-real-asset`
- `provenance_status = workspace-verified`
- `evidence_level = runtime-mainline`
- explicit `member_split_root` and `member_split_file`

So this lane does **not** require a new Local-API / Platform field or a new manifest schema bump.

### 3. The remaining work is boundary freezing, not execution release

What still needed maintenance was not code execution but carry-forward wording:

- `system-intake-ready` must not be misread as `paper-aligned`
- admitted gray-box rows must keep the exact provenance blocker visible
- future reopen must require real upstream identity change or protocol-aligned split/model assets, not wording drift

## Verdict

- `x26_pia_provenance_maintenance_main_lane_review = positive`

More precise reading:

1. the current `PIA` provenance blocker remains a real long-term blocker;
2. the blocker no longer needs new consumer fields or a new GPU release;
3. the correct outcome is to freeze the carry-forward boundary, intake reading, and reopen trigger explicitly.

## Frozen Posture

- `active_gpu_question = none`
- `next_gpu_candidate = none`
- `completed_task = X-26 PIA provenance maintenance main-lane review after X-25 reselection`
- `next_live_cpu_first_lane = X-27 non-graybox next-lane reselection after X-26 provenance review`
- `carry_forward_cpu_sidecar = I-A higher-layer boundary maintenance`

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-x26-pia-provenance-maintenance-main-lane-review.md`

## Handoff Decision

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`: update required
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`: update required
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`: update required
- `<DIFFAUDIT_ROOT>/Research/docs/mainline-narrative.md`: update required
- `<DIFFAUDIT_ROOT>/ROADMAP.md`: update required
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this pass freezes provenance boundary reading and consumer interpretation, but it does not add new exported fields, runtime requirements, or admitted metrics.
