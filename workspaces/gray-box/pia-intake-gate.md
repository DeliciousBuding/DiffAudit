# PIA Intake Gate

## Goal

This note defines when the gray-box `PIA` line is ready to be consumed as a **system next-step input**, without over-claiming paper alignment.

It is intentionally stricter than:

- `code exists`
- `smoke passed`

and intentionally weaker than:

- `paper-aligned reproduction ready`

## Canonical Roots

For the current `PIA` line, the canonical local asset roots are:

- checkpoint root: `workspaces/gray-box/assets/pia/checkpoints/cifar10_ddpm`
- dataset root: `workspaces/gray-box/assets/pia/datasets`
- source stash: `workspaces/gray-box/assets/pia/sources`
- member split root: `external/PIA/DDPM`
- latest verified run root: `workspaces/gray-box/runs/pia-cifar10-graybox-assets-probe-20260407`

## Current Canonical Status

The current canonical state of the `PIA` line is:

- runtime readiness: `single-machine ready`
- provenance: `workspace-verified`
- external split dependency: `external/PIA/DDPM/CIFAR10_train_ratio0.5.npz`

Interpretation:

- one verified machine can load the canonical checkpoint, dataset root, and external split dependency
- the historical next-run gate hashed the config and member split inputs and once re-checked `external/PIA` in a clean state
- the line is safe to consume as the next local research input
- provenance is now `workspace-verified`, so only `paper-aligned` remains blocked
- `paper-aligned` still remains blocked not only on release/source identity, but also on the CIFAR10 random-four-split / four-model protocol gap relative to the current single fixed split
- `system-intake-ready` does **not** imply `paper-aligned` or `benchmark-ready`

## Gate Levels

### 1. `template-blocked`

Meaning:

- shared template config still points at placeholders
- planner and repo layout may be valid
- real asset intake is not yet usable

Current evidence:

- [2026-04-07-pia-followup.md](2026-04-07-pia-followup.md)

### 2. `single-machine-asset-probe-ready`

Meaning:

- canonical local checkpoint path exists
- canonical local dataset root exists
- canonical local dataset layout exists
- external member split exists
- `probe-pia-assets`, `dry-run-pia`, and `runtime-probe-pia` all return `ready`

Current evidence:

- [2026-04-07-pia-real-asset-probe.md](2026-04-07-pia-real-asset-probe.md)
- [probe.json](runs/pia-cifar10-graybox-assets-probe-20260407/probe.json)
- [dry-run.json](runs/pia-cifar10-graybox-assets-probe-20260407/dry-run.json)
- [runtime-probe.json](runs/pia-cifar10-graybox-assets-probe-20260407/runtime-probe.json)

### 3. `single-machine-runtime-preview-ready`

Meaning:

- all asset-probe conditions above hold
- `runtime-preview-pia` can load real member and non-member CIFAR10 batches
- the adapter can produce preview scores on those real batches

Current evidence:

- [runtime-preview.json](runs/pia-cifar10-graybox-assets-probe-20260407/runtime-preview.json)

### 4. `system-intake-ready`

Meaning:

- the line is safe to consume as a **next-step local research input**
- the system may route follow-up work against this canonical asset root
- the line is **not** yet allowed to claim paper alignment or benchmark readiness

This level requires:

- `single-machine-runtime-preview-ready`
- explicit canonical roots
- explicit external split dependency
- explicit provenance status field
- explicit boundary note about what the current evidence does and does not prove

## Provenance Gate

The checkpoint provenance gate must be tracked separately from runtime readiness.

Allowed statuses:

- `source-retained-unverified`
- `workspace-verified`
- `paper-aligned`

Current status:

- `workspace-verified`

Reason:

- the checkpoint can be loaded and used by the current adapter
- the original source bundle is retained
- a machine-readable next-run gate exists
- the `2026-04-09` strict next-run gate passed under a clean `external/PIA` snapshot
- the current repo state must still be re-checked before any future strict release review
- paper alignment has not yet been confirmed
- the blocker only reopens on real upstream identity change or protocol-aligned split/model assets, not on wording drift

Implementation detail:

- the current local intake state now corresponds to `workspace-verified`

## External Split Boundary

The current `PIA` intake still depends on an external split file:

- `external/PIA/DDPM/CIFAR10_train_ratio0.5.npz`

That split is acceptable for the current local intake gate, but it remains an explicit external dependency.

So the current line is:

- `single-machine ready`
- `provenance workspace-verified`
- `external split retained`

## Runtime Preview Boundary

`runtime-preview-pia` currently proves only this:

- canonical local assets can be loaded
- real CIFAR10 member/non-member batches can be consumed
- the current adapter can produce preview scores on CPU

It does **not** prove:

- benchmark quality
- paper-faithful protocol alignment
- team-wide asset closure

## Handoff Checklist

Heisenberg / Laplace can treat `PIA` as the next-step local intake only if all items below are true:

- [ ] use the canonical asset roots listed in this document
- [ ] read the later same-day note [2026-04-07-pia-real-asset-probe.md](2026-04-07-pia-real-asset-probe.md), not only the earlier blocked template note
- [ ] confirm [probe.json](runs/pia-cifar10-graybox-assets-probe-20260407/probe.json) has `status = ready`
- [ ] confirm [runtime-probe.json](runs/pia-cifar10-graybox-assets-probe-20260407/runtime-probe.json) has `status = ready`
- [ ] confirm [runtime-preview.json](runs/pia-cifar10-graybox-assets-probe-20260407/runtime-preview.json) has `status = ready`
- [ ] carry forward provenance as `workspace-verified`
- [ ] carry forward the external split dependency explicitly as `external/PIA/DDPM/CIFAR10_train_ratio0.5.npz`
- [ ] do not upgrade any output to `paper-aligned` without separate provenance confirmation

## Next Acceptable System Step

The next acceptable system step is:

- build a real `PIA` runtime/mainline wrapper that consumes these canonical roots

The next unacceptable step is:

- presenting the current line as paper-aligned or benchmark-ready
