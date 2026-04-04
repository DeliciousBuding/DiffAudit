# Contributing

DiffAudit is organized as a multi-person research repository. The goal is to let multiple collaborators work in parallel without mixing paper reading, experiment planning, shared code, and local scratch work.

## Work Areas

Use the workspace that matches your current responsibility:

- `workspaces/black-box/`
- `workspaces/white-box/`
- `workspaces/gray-box/`
- `workspaces/implementation/`

These folders are for:

- reading notes
- reproduction checklists
- ownership notes
- experiment plans
- result summaries

Shared code belongs in:

- `src/diffaudit/`
- `configs/`
- `tests/`
- `scripts/`

## Branch and Commit Discipline

Use short focused commits.

Recommended branch naming:

- `black-box/<topic>`
- `white-box/<topic>`
- `gray-box/<topic>`
- `implementation/<topic>`

Recommended commit style:

- `feat: ...`
- `fix: ...`
- `chore: ...`
- `docs: ...`
- `test: ...`

## Experiment Naming

Configs should be explicit and stable.

Recommended naming:

- `configs/attacks/<method>_<purpose>.yaml`
- `configs/benchmarks/<method>_<purpose>.yaml`

Experiment outputs should be written under:

- `experiments/<run-name>/`

Use run names that encode method and purpose, for example:

- `secmi-smoke`
- `secmi-cifar10-blackbox`

## Asset Handling

Do not commit private datasets, checkpoints, or local scratch clones.

Track assets by path in configs, but keep the assets themselves outside git unless they are intentionally mirrored research materials under `references/materials/`.

If required assets are missing:

- implement a `dry-run` or `blocked` path
- record the missing path clearly
- do not claim the experiment is runnable

## Third-Party Code

Use `third_party/` only for minimal vendored subsets required for integration.

Rules:

- keep the subset small
- preserve attribution
- patch only what is needed for integration or portability
- do not commit exploratory upstream clones from `external/`

## Testing

Before claiming a change is complete, run the relevant tests.

Current baseline:

```powershell
$env:PYTHONPATH='src;.'
python -m unittest
```

## Public Repository Notes

This repository is intended to be publishable as a public GitHub repository.

Before adding large files:

- check total size
- check per-file size
- update `references/materials/README.md` if mirrored materials change
