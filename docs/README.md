# Documentation Index

This directory contains stable project documentation for `Research/`. It should not become a dump for one-off prompts, private logs, or local machine instructions.

## First Reading Path

| Step | Document | Purpose |
| --- | --- | --- |
| 1 | [../README.md](../README.md) | Repository front door and quick orientation |
| 2 | [teammate-setup.md](teammate-setup.md) | New-machine setup and first validation commands |
| 3 | [data-and-assets-handoff.md](data-and-assets-handoff.md) | How to obtain the same datasets, weights, and supplementary bundles |
| 4 | [command-reference.md](command-reference.md) | Runnable CLI recipes by track |
| 5 | [comprehensive-progress.md](comprehensive-progress.md) | One-page current research progress |
| 6 | [reproduction-status.md](reproduction-status.md) | Per-track reproduction and evidence state |
| 7 | [mainline-narrative.md](mainline-narrative.md) | Claim boundaries and presentation-facing research story |

## Core Docs

| Document | Purpose |
| --- | --- |
| [environment.md](environment.md) | Conda, CUDA, and package setup |
| [storage-boundary.md](storage-boundary.md) | Where raw downloads, code clones, workspace assets, and evidence belong |
| [download-naming-policy.md](download-naming-policy.md) | Naming rules for `<DIFFAUDIT_ROOT>/Download/` |
| [research-download-master-list.md](research-download-master-list.md) | Rebuild list for first-wave datasets, weights, and supplementary bundles |
| [repo-map.md](repo-map.md) | Directory and code responsibility map |
| [getting-started.md](getting-started.md) | Short onboarding guide for contributors |
| [github-collaboration.md](github-collaboration.md) | Branch and PR collaboration rules |
| [licensing.md](licensing.md) | Project license scope and third-party material boundaries |
| [researcher-agent-architecture.md](researcher-agent-architecture.md) | Long-running ResearcherAgent operating model |
| [research-autonomous-execution-prompt.md](research-autonomous-execution-prompt.md) | Current autonomous execution prompt surface |

## Research Status And Planning

| Document | Purpose |
| --- | --- |
| [admitted-results-summary.md](admitted-results-summary.md) | Human-readable admitted result summary |
| [future-phase-e-intake.md](future-phase-e-intake.md) | Candidate intake queue and entry gates |
| [next-run-intake-index.md](next-run-intake-index.md) | Next-run entrypoint index and contracts |
| [mentor-strict-reproduction-plan.md](mentor-strict-reproduction-plan.md) | Strict parallel reproduction plan |
| [mia-defense-research-index.md](mia-defense-research-index.md) | Research index for the MIA defense strategy |
| [mia-defense-execution-checklist.md](mia-defense-execution-checklist.md) | Action checklist for the MIA defense strategy |
| [innovation-evidence-map.md](innovation-evidence-map.md) | Evidence map for innovation claims |

## Runtime And System Boundaries

| Document | Purpose |
| --- | --- |
| [runtime.md](runtime.md) | Research-facing notes for the sibling `Runtime-Server/` service |
| [local-api.md](local-api.md) | Historical Python compatibility layer notes |
| [asset-registry-local-api.md](asset-registry-local-api.md) | Research-to-Runtime registry-alignment contract |
| [recon-artifact-replay-guidance.md](recon-artifact-replay-guidance.md) | How to interpret recon replay/debug traces |
| [research-boundary-card.md](research-boundary-card.md) | Short boundary card for cross-repo consumers |

## Literature And Reports

| Location | Purpose |
| --- | --- |
| [paper-reports/README.md](paper-reports/README.md) | Paper report package and report-generation rules |
| [report-bundles/README.md](report-bundles/README.md) | External model result bundles retained for review |
| [recon-public-asset-mapping.md](recon-public-asset-mapping.md) | Public recon asset mapping and semantic boundary |

## Maintenance Rules

- Keep the root README short and reader-facing.
- Put command catalogues in [command-reference.md](command-reference.md), not in the root README.
- Put data and path setup in [data-and-assets-handoff.md](data-and-assets-handoff.md) and ignored local config files.
- Put current status in [comprehensive-progress.md](comprehensive-progress.md), [reproduction-status.md](reproduction-status.md), and `ROADMAP.md`.
- Use `docs/` for stable documentation only. Temporary prompts, scratch context, and local operator logs should move to a workspace note or an ignored local file.
