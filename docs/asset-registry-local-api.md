# Asset Registry Intake Contract (Local-API)

This document defines the smallest contract surface for integrating:

- the **Local-API SQLite registry + runner system** (`Services/Local-API`)
- the **Research-side machine-readable asset metadata** (`Research/workspaces/intake`)

Scope boundary:

- This is **not** a Platform design doc.
- This is **not** a full asset hosting strategy.
- It only standardizes the minimal, machine-checkable fields needed for Local-API to safely consume promoted assets.

## Canonical Machine-Readable Entry Points

Research-side (source of truth for assets/intake):

- `Research/workspaces/intake/index.json` (`diffaudit.intake.index.v1`)
- `Research/workspaces/**/assets/**/manifest*.json` (`diffaudit.intake.manifest.v1`)
- `Research/workspaces/implementation/artifacts/unified-attack-defense-table.json` (`diffaudit.attack_defense_table.v1`) as the admitted cross-track result table

Local-API-side (source of truth for contract registry / job routing):

- `Services/Local-API/internal/api/registry_seed.json` (seed payload)
- `Services/Local-API/internal/api/registry_store.go` (SQLite schema for `contracts` + `jobs`)

## Admitted Display Contract

Platform / Local-API consuming the admitted results surface MUST treat:

- `Research/workspaces/implementation/artifacts/unified-attack-defense-table.json`

as the only machine-readable headline table for admitted cross-track results.

Every admitted row exposed to downstream consumers MUST carry:

- `track`
- `attack`
- `defense`
- `model`
- `auc`
- `asr`
- `evidence_level`
- `note`
- `source`
- `boundary`

If a UI or API wants to expose non-admitted directions, it MUST label them as exactly one of:

- `comparator`
- `blocked baseline`
- `hold`
- `intake only`

and MUST NOT render them at the same headline level as admitted rows.

## Minimal Intake Manifest Fields (Required)

Every manifest referenced by `Research/workspaces/intake/index.json` MUST contain:

- `schema`: must be `diffaudit.intake.manifest.v1`
- `contract_key`: must equal `entries[i].contract_key`
- `track`: must equal `entries[i].track`
- `method`: must equal `entries[i].method`
- `contract_stage`: current contract stage from research POV (e.g. `target`)
- `asset_grade`: asset completeness / locality grade
- `provenance_status`: provenance classification for promotion decisions
- `evidence_level`: best current evidence level (e.g. `runtime-mainline`)

Current boundary note:

- `Research/workspaces/intake/index.json` currently covers promoted intake contracts, not every admitted research result.
- `Research/workspaces/intake/index.json.entries[]` is the only Research-side surface covered by the Local-API intake contract.
- `Research/workspaces/intake/phase-e-candidates.json` is a research-owned Phase E candidate ordering supplement and is not part of Local-API job routing.
- `PIA` has a stable promoted intake surface.
- `GSA` currently has a legacy intake surface plus a stronger admitted result tracked in the unified table and white-box docs.
- `recon` and `DPDM` are currently admitted through the unified table plus frozen workspace docs, not standalone intake manifests.
- New external zip / checkpoint inputs are quarantine-first assets: they require a Research admission judgment before they can be referenced anywhere outside intake or reference-only records.

Candidate-boundary hard rule:

- `phase-e-candidates.json` records MUST NOT define `contract_key`.
- `phase-e-candidates.json` records MUST NOT define `manifest`.
- `phase-e-candidates.json` records MUST NOT define `compatibility.commands`.
- `phase-e-candidates.json` records MUST NOT be read as admitted, benchmark-ready, or execution-ready contracts.

Method-specific required fields MUST be declared in the intake index:

- `entries[i].compatibility.commands[j].required_manifest_fields`

And MUST exist in the referenced manifest. This is the minimal bridge that lets Local-API (later) validate asset readiness per job surface without hardcoding method-specific layouts.

## Path Semantics (Hard Rule)

All path strings in:

- `Research/workspaces/intake/index.json`
- intake manifests referenced by that index
- `Research/workspaces/implementation/artifacts/unified-attack-defense-table.json` row `source` fields

MUST be **repo-relative paths**, so the intake registry stays portable across machines. (Untracked directories like `external/**` are allowed as long as they are still repo-relative.)

Local-API resolves these paths by joining them with `research_root` at runtime.

## Local-API Registry Alignment (What Must Match)

For each `contract_key` that is represented in `index.json.entries[]`:

1. `Services/Local-API/internal/api/registry_seed.json` MUST contain that `contract_key`.
2. The Local-API `contracts.promoted_asset_roots_json` MUST point under the intake entry `paths.assets_root`.

Note: the registry seed currently uses `Research/workspaces/...`-prefixed roots while the intake index uses `workspaces/...`. Alignment is validated by treating `Research/` as an optional prefix for promoted roots.

## Migration Order (Minimal, Enforceable)

When onboarding a new promoted asset set (or migrating an existing one):

1. Put the assets under `Research/workspaces/<track>/assets/<method>/...` (or another line-owned subtree).
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
python Research/scripts/validate_intake_index.py
python Research/scripts/validate_local_api_registry_alignment.py
```

