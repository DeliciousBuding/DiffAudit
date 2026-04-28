# Contributing To DiffAudit Research

Thank you for improving DiffAudit Research. This repository contains the
research, reproducibility, and evidence-production side of DiffAudit. Good
contributions make privacy-risk claims easier to inspect, reproduce, and hand
off to the wider DiffAudit system.

## Contribution Areas

| Area | Typical changes |
| --- | --- |
| Research lanes | Paper intake, bounded probes, experiment plans, evidence anchors |
| Reproducibility | Environment setup, asset contracts, command recipes, smoke checks |
| Implementation | Python package code, CLI commands, adapters, tests, scripts |
| Documentation | Setup guides, evidence status, claim boundaries, licensing notes |
| Governance | GitHub templates, CI, security, citation, contribution workflow |

## Branches And Pull Requests

Create a branch from `main` and open a pull request for shared changes. Useful
branch prefixes include:

| Prefix | Use for |
| --- | --- |
| `black-box/` | Black-box attack or evidence work |
| `gray-box/` | Gray-box attack, defense, or provenance work |
| `white-box/` | White-box attack, defense, or bridge work |
| `implementation/` | Shared code, CLI, config, tests, or scripts |
| `docs/` | Documentation and repository surface polish |
| `chore/` | CI, metadata, dependency, or maintenance updates |

Pull requests should explain:

| Field | What to include |
| --- | --- |
| Summary | What changed |
| Why | The research, reproducibility, or repository problem addressed |
| Validation | Commands run, checks skipped, and why |
| Claim boundary | What the change does not prove |
| Follow-ups | Known blockers or next steps |

## Commit Style

Use concise conventional prefixes when possible:

```text
feat: add a bounded probe
fix: correct an asset path renderer
docs: clarify reproduction status
test: cover a CLI contract
chore: update repository metadata
```

Keep commits focused. Avoid mixing unrelated paper notes, code refactors,
generated outputs, and documentation changes in the same commit.

## Research Evidence Rules

DiffAudit distinguishes engineering readiness from research evidence. When a
result changes status, update the relevant evidence anchor and status document.

Use the canonical reproduction ladder from
[docs/reproduction-status.md](docs/reproduction-status.md):

| State | Meaning |
| --- | --- |
| `research-ready` | Papers, upstream code, and asset requirements have been reviewed. |
| `code-ready` | Commands, configs, and tests are present. |
| `evidence-ready` | The repository contains a submit-ready summary or equivalent run evidence. |
| `asset-ready` | Required datasets, weights, or supplementary files are available and probed. |
| `benchmark-ready` | The lane can reasonably claim paper-level benchmark execution or recomputation. |

Do not present dry-runs, smoke tests, or synthetic checks as benchmark claims.
If a hypothesis fails, record the negative or blocked verdict instead of
dropping the result.

## Assets And Third-Party Material

Do not commit private datasets, gated model weights, credentials, large raw
assets, or local external clones. Use the asset handoff documents instead:

| Document | Purpose |
| --- | --- |
| [docs/data-and-assets-handoff.md](docs/data-and-assets-handoff.md) | How to obtain and bind datasets, weights, and supplementary bundles |
| [docs/download-naming-policy.md](docs/download-naming-policy.md) | Naming policy for the project asset mirror |
| [docs/research-download-master-list.md](docs/research-download-master-list.md) | Rebuild list for first-wave research assets |
| [docs/licensing.md](docs/licensing.md) | License scope and third-party material boundaries |

`third_party/` is reserved for minimal vendored upstream subsets with retained
notices. Ignored external code clones and large assets should stay outside the
tracked repository.

## Validation

Run the standard local gate before opening or updating a PR:

```powershell
python scripts/run_local_checks.py
```

For smaller documentation-only changes, run at least:

```powershell
git diff --check
python -m diffaudit --help
```

The GitHub Actions baseline runs the CLI install check, local config rendering,
and core tests on Windows.

## Review Expectations

Reviewers should focus on:

| Risk | What to check |
| --- | --- |
| Reproducibility drift | Commands, configs, assets, manifests, and evidence anchors still align |
| Claim inflation | The PR does not overstate smoke, dry-run, or partial results |
| Path safety | Shared docs and configs do not depend on private machine paths |
| Licensing | Third-party code, papers, datasets, and weights retain their own terms |
| Tests | Behavior changes include focused tests or a documented reason for omission |

Copilot or other automated reviewers can be useful as a first pass, but final
acceptance should be based on repository evidence and human review.

## Security

Do not include secrets, private asset links, credentials, proprietary datasets,
or unpublished model weights in public issues, pull requests, screenshots, or
logs. See [SECURITY.md](SECURITY.md) for private reporting guidance.
