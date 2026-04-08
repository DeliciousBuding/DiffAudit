# Asset Registry Intake Contract (Local-API)

This document defines the smallest contract surface for integrating:

- the **Local-API SQLite registry + runner system** (`Services/Local-API`)
- the **Project-side machine-readable asset metadata** (`Project/workspaces/intake`)

Scope boundary:

- This is **not** a Platform design doc.
- This is **not** a full asset hosting strategy.
- It only standardizes the minimal, machine-checkable fields needed for Local-API to safely consume promoted assets.

## Canonical Machine-Readable Entry Points

Project-side (source of truth for assets/intake):

- `Project/workspaces/intake/index.json` (`diffaudit.intake.index.v1`)
- `Project/workspaces/**/assets/**/manifest*.json` (`diffaudit.intake.manifest.v1`)
- `Project/workspaces/implementation/artifacts/unified-attack-defense-table.json` (`diffaudit.attack_defense_table.v1`) as the admitted cross-track result table

Local-API-side (source of truth for contract registry / job routing):

- `Services/Local-API/internal/api/registry_seed.json` (seed payload)
- `Services/Local-API/internal/api/registry_store.go` (SQLite schema for `contracts` + `jobs`)

## Minimal Intake Manifest Fields (Required)

Every manifest referenced by `Project/workspaces/intake/index.json` MUST contain:

- `schema`: must be `diffaudit.intake.manifest.v1`
- `contract_key`: must equal `entries[i].contract_key`
- `track`: must equal `entries[i].track`
- `method`: must equal `entries[i].method`
- `contract_stage`: current contract stage from research POV (e.g. `target`)
- `asset_grade`: asset completeness / locality grade
- `provenance_status`: provenance classification for promotion decisions
- `evidence_level`: best current evidence level (e.g. `runtime-mainline`)

Current boundary note:

- `Project/workspaces/intake/index.json` currently covers promoted intake contracts, not every admitted research result.
- `PIA` has a stable promoted intake surface.
- `GSA` currently has a legacy intake surface plus a stronger admitted result tracked in the unified table and white-box docs.
- `recon` and `DPDM` are currently admitted through the unified table plus frozen workspace docs, not standalone intake manifests.

Method-specific required fields MUST be declared in the intake index:

- `entries[i].compatibility.commands[j].required_manifest_fields`

And MUST exist in the referenced manifest. This is the minimal bridge that lets Local-API (later) validate asset readiness per job surface without hardcoding method-specific layouts.

## Path Semantics (Hard Rule)

All path strings in:

- `Project/workspaces/intake/index.json`
- intake manifests referenced by that index

MUST be **repo-relative paths**, so the intake registry stays portable across machines. (Untracked directories like `external/**` are allowed as long as they are still repo-relative.)

Local-API resolves these paths by joining them with `project_root` at runtime.

## Local-API Registry Alignment (What Must Match)

For each `contract_key` that is represented in the intake index:

1. `Services/Local-API/internal/api/registry_seed.json` MUST contain that `contract_key`.
2. The Local-API `contracts.promoted_asset_roots_json` MUST point under the intake entry `paths.assets_root`.

Note: the registry seed currently uses `Project/workspaces/...`-prefixed roots while the intake index uses `workspaces/...`. Alignment is validated by treating `Project/` as an optional prefix for promoted roots.

## Migration Order (Minimal, Enforceable)

When onboarding a new promoted asset set (or migrating an existing one):

1. Put the assets under `Project/workspaces/<track>/assets/<method>/...` (or another line-owned subtree).
2. Write/update the manifest JSON and include the required common fields listed above.
3. Add/update the intake index entry:
   - `contract_key`, `track`, `method`
   - `paths.assets_root`
   - `manifest` path
   - `compatibility.commands[].required_manifest_fields`
4. Confirm the Local-API registry seed `promoted_asset_roots` all live under `paths.assets_root` for that contract.
5. Run the validation gates below and treat non-zero exit codes as a hard block.

## Validation (Must Pass Before Claiming "Ready")

From the workspace root:

```bash
python Project/scripts/validate_intake_index.py
python Project/scripts/validate_local_api_registry_alignment.py
```

