# 2026-04-23 Research Storage Cleanup Savepoint

## What Was Physically Changed

### 1. `recon-assets` moved out of `external/`

Old:

- `D:\Code\DiffAudit\Research\external\recon-assets\`

New canonical raw location:

- `D:\Code\DiffAudit\Download\black-box\supplementary\recon-assets\`

This is now the only intended live raw bundle location.

## What Was Clarified

### 2. `SecMI` split is now explicit

- full upstream reference clone:
  - `D:\Code\DiffAudit\Research\external\SecMI\`
- canonical minimal in-repo integration surface:
  - `D:\Code\DiffAudit\Research\third_party\secmi\`

### 3. `CLiD` split is now explicit

- working code clone:
  - `D:\Code\DiffAudit\Research\external\CLiD\`
- raw supplementary mirror:
  - `D:\Code\DiffAudit\Download\black-box\supplementary\clid-mia-supplementary\`

## New Boundary Entry Files

- `D:\Code\DiffAudit\Research\external\README.md`
- `D:\Code\DiffAudit\Research\third_party\README.md`
- `D:\Code\DiffAudit\Download\README.md`
- `D:\Code\DiffAudit\Research\external\SecMI\LOCAL_ROLE.md`
- `D:\Code\DiffAudit\Research\external\CLiD\LOCAL_ROLE.md`
- `D:\Code\DiffAudit\Research\third_party\secmi\LOCAL_ROLE.md`
- `D:\Code\DiffAudit\Research\docs\storage-boundary.md`

## Active Entry Docs Updated

- `D:\Code\DiffAudit\Research\README.md`
- `D:\Code\DiffAudit\Research\docs\teammate-setup.md`
- `D:\Code\DiffAudit\Research\docs\recon-public-asset-mapping.md`
- `D:\Code\DiffAudit\Research\docs\research-download-master-list.md`
- `D:\Code\DiffAudit\Research\workspaces\README.md`
- `D:\Code\DiffAudit\Research\ROADMAP.md`

## Residual Historical Drift

Some historical notes and legacy docs still preserve old path wording intentionally as historical context.

That is acceptable as long as:

- active entry docs use the new canonical paths
- no live command examples point back to `external/recon-assets`

## Current Canonical Rule

- upstream/exploratory code clones -> `Research/external/`
- minimal vendored integration -> `Research/third_party/`
- raw downloads / supplementary / checkpoints -> `D:\Code\DiffAudit\Download\`
- lane-normalized gateways -> `Research/workspaces/<lane>/assets/`
- run evidence -> `Research/workspaces/<lane>/runs/`

## Canonical Evidence Anchor

- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-23-research-storage-cleanup-savepoint.md`
