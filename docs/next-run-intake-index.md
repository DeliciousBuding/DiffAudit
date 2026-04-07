# Next-Run Intake Index

This page is an index of runnable "next-run" entrypoints and their intake/output contracts.
It is meant to be one-line attachable to `GLOBAL_TASK_BOARD` without pulling in per-track notebooks.

## GSA (white-box) next-run intake gate

Board one-liner:
`GSA next-run: tools/gsa_next_run emits manifest+provenance (input hashes + git + host) as the intake gate.`

Command (no install):

```powershell
cd Project/tools/gsa_next_run
.\run.ps1 --assets-root .\examples\minimal\assets_root --repo-root .\examples\minimal\repo_root --config .\examples\minimal\config --out-dir ..\..\tmp\gsa_next_run_smoke
```

Inputs:
- `--assets-root <dir>`: runtime assets directory (hashed as a tree)
- `--config <file|dir>`: run config (hashed by content)
- `--repo-root <dir>`: used to record git commit (and optionally enforce clean repo via `--strict`)

Outputs (under `--out-dir`):
- `manifest.json`: inputs + hashes + git + validation
- `provenance.json`: host + timestamp + manifest sha256 + validation

Exit codes:
- `0`: validation OK
- `2`: validation failed (intake blocked)

Reference:
- gate README: [../tools/gsa_next_run/README.md](../tools/gsa_next_run/README.md)

## PIA (gray-box) next-run runtime mainline

Board one-liner:
`PIA next-run: diffaudit run-pia-runtime-mainline consumes local canonical assets and writes summary+scores into workspace.`

Command (existing entrypoint; local config must point at your dataset/checkpoint roots):

```powershell
python -m diffaudit run-pia-runtime-mainline --config tmp/configs/pia-cifar10-graybox-assets.local.yaml --workspace workspaces/gray-box/runs/pia-next-run --repo-root external/PIA --member-split-root external/PIA/DDPM --device cpu
```

Outputs (under `--workspace`):
- `summary.json`
- `scores.json`

Reference:
- runtime evidence note: [../workspaces/gray-box/2026-04-07-pia-runtime-mainline.md](../workspaces/gray-box/2026-04-07-pia-runtime-mainline.md)

