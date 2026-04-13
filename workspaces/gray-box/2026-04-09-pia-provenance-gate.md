# 2026-04-09 Gray-Box Follow-Up: PIA Provenance Gate

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-09 00:13:46 +08:00`
- `selected_mainline`: `PIA`
- `current_state`: `strict next-run provenance gate passed`
- `evidence_level`: `intake-gate`

## Commands Run

Relaxed gate:

```powershell
powershell -ExecutionPolicy Bypass -File tools/pia_next_run/run.ps1 `
  --config tmp/configs/pia-cifar10-graybox-assets.local.yaml `
  --member-split-root external/PIA/DDPM `
  --repo-root external/PIA `
  --out-dir workspaces/gray-box/assets/pia/next-run-20260409
```

Strict gate:

```powershell
powershell -ExecutionPolicy Bypass -File tools/pia_next_run/run.ps1 `
  --config tmp/configs/pia-cifar10-graybox-assets.local.yaml `
  --member-split-root external/PIA/DDPM `
  --repo-root external/PIA `
  --out-dir workspaces/gray-box/assets/pia/next-run-20260409-strict `
  --strict
```

## Outputs

- relaxed manifest:
  - `workspaces/gray-box/assets/pia/next-run-20260409/manifest.json`
- relaxed provenance:
  - `workspaces/gray-box/assets/pia/next-run-20260409/provenance.json`
- strict manifest:
  - `workspaces/gray-box/assets/pia/next-run-20260409-strict/manifest.json`
- strict provenance:
  - `workspaces/gray-box/assets/pia/next-run-20260409-strict/provenance.json`

## Result

- relaxed gate:
  - `validation.ok = true`
  - warning:
    - `repo_root has uncommitted changes (git status not clean)`
- strict gate:
  - `validation.ok = true`
  - warnings:
    - none
  - `manifest_sha256 = 6ca897687d0e84a0d13b27c6ed4767a25a8eb24c3e676b115ac5658cdd7fff37`
  - `git.commit = 0d7e08a5a07f44931692d52d54d0ce41aff8f54c`
  - `git.dirty = false`

## Interpretation

- The gray-box `PIA` line obtained a clean machine-readable next-run provenance gate on `2026-04-09`.
- The config and member split inputs are hashed.
- The referenced `external/PIA` repo was clean under that strict validation snapshot.
- This is enough to lock the local asset line at `workspace-verified`.
- It is still not enough to promote the line to `paper-aligned`, because release identity, source-bundle version/checksum alignment, `external/PIA` upstream identity, and split semantics remain unverified under the closed provenance dossier.
- This gate is not a release signal for `PIA paper-aligned confirmation`; until provenance materially changes, execution-layer status remains `no-go`.

## Current-Time Caveat

This note is a historical gate record, not a present-tense cleanliness claim.

As of `2026-04-10 round-29`, a new strict redo at:

- `workspaces/gray-box/assets/pia/next-run-20260410-round29-strict-redo`

returned:

- `validation.ok = false`
- `error = repo_root is dirty`

Therefore:

- `2026-04-09 strict gate passed` remains valid historical evidence
- but `external/PIA is currently clean` is no longer an allowed statement
