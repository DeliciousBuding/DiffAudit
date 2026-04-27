# 2026-04-23 Research Storage Cleanup Savepoint

## What Was Physically Changed

### 1. `recon-assets` moved out of `external/`

Old:

- `<DIFFAUDIT_ROOT>/Research/external/recon-assets\`

New canonical raw location:

- `<DIFFAUDIT_ROOT>/Download/black-box/supplementary/recon-assets\`

This is now the only intended live raw bundle location.

## What Was Clarified

### 2. `SecMI` split is now explicit

- full upstream reference clone:
  - `<DIFFAUDIT_ROOT>/Research/external/SecMI\`
- canonical minimal in-repo integration surface:
  - `<DIFFAUDIT_ROOT>/Research/third_party/secmi\`

### 3. `CLiD` split is now explicit

- working code clone:
  - `<DIFFAUDIT_ROOT>/Research/external/CLiD\`
- raw supplementary mirror:
  - `<DIFFAUDIT_ROOT>/Download/black-box/supplementary/clid-mia-supplementary\`

## New Boundary Entry Files

- `<DIFFAUDIT_ROOT>/Research/external/README.md`
- `<DIFFAUDIT_ROOT>/Research/third_party/README.md`
- `<DIFFAUDIT_ROOT>/Download/README.md`
- `<DIFFAUDIT_ROOT>/Research/external/SecMI/LOCAL_ROLE.md`
- `<DIFFAUDIT_ROOT>/Research/external/CLiD/LOCAL_ROLE.md`
- `<DIFFAUDIT_ROOT>/Research/third_party/secmi/LOCAL_ROLE.md`
- `<DIFFAUDIT_ROOT>/Research/docs/storage-boundary.md`

## Active Entry Docs Updated

- `<DIFFAUDIT_ROOT>/Research/README.md`
- `<DIFFAUDIT_ROOT>/Research/docs/teammate-setup.md`
- `<DIFFAUDIT_ROOT>/Research/docs/recon-public-asset-mapping.md`
- `<DIFFAUDIT_ROOT>/Research/docs/research-download-master-list.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/README.md`
- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`

## Residual Historical Drift

Some historical notes and legacy docs still preserve old path wording intentionally as historical context.

That is acceptable as long as:

- active entry docs use the new canonical paths
- no live command examples point back to `external/recon-assets`

## Current Canonical Rule

- upstream/exploratory code clones -> `Research/external/`
- minimal vendored integration -> `Research/third_party/`
- raw downloads / supplementary / checkpoints -> `<DIFFAUDIT_ROOT>/Download/`
- lane-normalized gateways -> `Research/workspaces/<lane>/assets/`
- run evidence -> `Research/workspaces/<lane>/runs/`

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-23-research-storage-cleanup-savepoint.md`
