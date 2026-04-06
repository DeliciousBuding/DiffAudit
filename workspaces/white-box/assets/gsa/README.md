# GSA Assets Root

This directory is the formal local asset root for the white-box `GSA` line.

It exists to keep three concerns separate:

- `external/GSA`
  - upstream code clone
- `workspaces/white-box/assets/gsa`
  - local-only assets, manifests, and source bundles
- `workspaces/white-box/runs`
  - execution outputs and summaries

## Intended Subtrees

Expected local layout:

- `datasets/`
  - extracted datasets and split-ready image buckets
- `checkpoints/`
  - only `GSA`-compatible training outputs, especially `checkpoint-*` directories
- `manifests/`
  - provenance notes, split manifests, ownership notes
- `sources/`
  - raw archives or import bundles

These assets are local-only and are ignored by git, except for this documentation.

## Boundary Rules

Use this root when the material is:

- required by the white-box `GSA` line
- too large or too local to commit
- not an execution output

Do **not** use this root for:

- upstream code
- toy smoke outputs
- gray-box `PIA` assets

Those belong elsewhere:

- code: `workspaces/white-box/external/GSA`
- runs: `workspaces/white-box/runs`
- gray-box assets: `workspaces/gray-box/assets/pia`

## Current Status

This root is now official, but it does not yet contain paper-aligned `GSA` assets.

Current blocker remains:

- no `GSA`-compatible `checkpoint-*` directory state

For operational next steps, read [HANDOFF.md](HANDOFF.md).
