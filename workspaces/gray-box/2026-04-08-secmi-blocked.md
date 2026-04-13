# 2026-04-08 Gray-Box Follow-Up: SecMI Blocked Baseline

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-08 23:17:22 +08:00`
- `selected_baseline`: `SecMI`
- `current_state`: `blocked baseline`
- `evidence_level`: `runtime-probe`

## Command Run

```powershell
conda run -n diffaudit-research python -m diffaudit runtime-probe-secmi `
  --config tmp/configs/rendered-checks/secmi.local.yaml `
  --repo-root external/SecMI
```

## Probe Result

```json
{
  "status": "blocked",
  "error": "Missing SecMI flagfile: REPLACE_WITH_SECMI_MODEL_DIR\\flagfile.txt",
  "repo_root": "external/SecMI"
}
```

## Interpretation

- The current repository still only has a placeholder-style local config for `SecMI`.
- The first hard blocker is missing real `flagfile.txt` and matching model root layout.
- This is an asset/layout blocker, not a GPU blocker.

## Current Decision

`SecMI` is retained as a `blocked baseline` for the gray-box line.

It stays in the roadmap and comparison narrative, but it does not compete with `PIA` for current GPU priority until a real checkpoint root and flagfile are available.
