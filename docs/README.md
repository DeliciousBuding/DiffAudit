# DiffAudit Research Documentation

This directory is the documentation map for DiffAudit Research. It separates
public-facing setup, reproducibility, asset, and evidence documents from deeper
research planning and operating references.

## Public Surface Boundary

The root [README](../README.md), this documentation map, and the documents in
the Public Onboarding, Setup And Reproducibility, Collaboration And Governance,
and brand/licensing/security entries are the public front door for external
readers.

Internal planning artifacts may still live under `docs/`, but they should not
be presented as product copy or onboarding material unless they have been
rewritten for that audience. In particular, raw agent prompts, deadline-specific
notes, review dumps, and operator-local instructions must remain separate from
the public reading path.

## Public Onboarding Path

| Step | Document | Purpose |
| --- | --- | --- |
| 1 | [../README.md](../README.md) | Repository front door and quick orientation |
| 2 | [getting-started.md](getting-started.md) | Short onboarding guide for contributors |
| 3 | [teammate-setup.md](teammate-setup.md) | New-machine setup and first validation commands |
| 4 | [data-and-assets-handoff.md](data-and-assets-handoff.md) | How to obtain the same datasets, weights, and supplementary bundles |
| 5 | [command-reference.md](command-reference.md) | Runnable CLI recipes by track |
| 6 | [reproduction-status.md](reproduction-status.md) | Per-track reproduction and evidence state |
| 7 | [admitted-results-summary.md](admitted-results-summary.md) | Human-readable admitted result summary |

## Setup And Reproducibility

| Document | Purpose |
| --- | --- |
| [environment.md](environment.md) | Conda, CUDA, and package setup |
| [storage-boundary.md](storage-boundary.md) | Where raw downloads, code clones, workspace assets, and evidence belong |
| [download-naming-policy.md](download-naming-policy.md) | Naming rules for `<DIFFAUDIT_ROOT>/Download/` |
| [research-download-master-list.md](research-download-master-list.md) | Rebuild list for first-wave datasets, weights, and supplementary bundles |
| [repo-map.md](repo-map.md) | Directory and code responsibility map |
| [getting-started.md](getting-started.md) | Short onboarding guide for contributors |
| [licensing.md](licensing.md) | Project license scope and third-party material boundaries |
| [brand-assets.md](brand-assets.md) | Logo assets and README image-hosting policy |

## Research Status And Planning

| Document | Purpose |
| --- | --- |
| [comprehensive-progress.md](comprehensive-progress.md) | Active research handoff and current progress ledger; not product copy |
| [mainline-narrative.md](mainline-narrative.md) | Active research narrative draft and claim-boundary notes; extract public copy before reuse |
| [future-phase-e-intake.md](future-phase-e-intake.md) | Candidate intake queue and entry gates |
| [next-run-intake-index.md](next-run-intake-index.md) | Next-run entrypoint index and contracts |
| [mentor-strict-reproduction-plan.md](mentor-strict-reproduction-plan.md) | Strict parallel reproduction plan |
| [mia-defense-research-index.md](mia-defense-research-index.md) | Research index for the MIA defense strategy |
| [mia-defense-execution-checklist.md](mia-defense-execution-checklist.md) | Action checklist for the MIA defense strategy |
| [innovation-evidence-map.md](innovation-evidence-map.md) | Evidence map for innovation claims |

## Collaboration And Governance

| Document | Purpose |
| --- | --- |
| [github-collaboration.md](github-collaboration.md) | Branch, permission, and PR collaboration guide |
| [github-settings-baseline.md](github-settings-baseline.md) | GitHub repository setting baseline |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Contribution workflow |
| [../SECURITY.md](../SECURITY.md) | Security reporting policy and scope |
| [../CITATION.cff](../CITATION.cff) | Machine-readable repository citation metadata |

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

The root [README](../README.md) remains the public front door. Detailed command
catalogues, asset setup, and current evidence status live in the linked docs
above so the landing page can stay concise.
