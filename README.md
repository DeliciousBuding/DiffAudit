# DiffAudit

DiffAudit is a research-oriented privacy auditing system for diffusion models.

The project does not aim to improve image generation quality. Its purpose is to study whether diffusion models memorize training samples too strongly, to reproduce membership inference attacks, and to turn attack evidence into an audit-oriented workflow.

## Research Scope

DiffAudit currently tracks three attack directions:

- `black-box`: realistic API-facing membership inference
- `white-box`: strongest attack setting with internal model access
- `gray-box`: partial internal visibility with stronger explanation signals

Current repository priority is `black-box`. White-box and gray-box work should extend the same config, adapter, and result-recording structure instead of branching into separate ad hoc code paths.

## Collaboration Model

This is a multi-person research repository.

Collaboration is organized around workspaces:

- `workspaces/black-box/`: black-box paper intake, configs, notes, and experiment planning
- `workspaces/white-box/`: white-box paper intake and reproduction work
- `workspaces/gray-box/`: gray-box or semi-white-box work
- `workspaces/implementation/`: shared engineering tasks such as adapters, runners, reporting, and infrastructure

Shared production code still lives under `src/diffaudit/`, while each workspace is used to organize ownership, notes, and experiment planning without mixing everyone’s work into the same folder.

## Current Status

The repository already includes:

- an isolated GPU-ready research environment
- config-driven smoke pipelines
- `SecMI` planning, artifact resolution, workspace validation, adapter preparation, and dry-run validation
- a vendored minimal `SecMI` attack subset under `third_party/secmi`
- reproducibility-oriented unit tests

The repository does not yet include:

- checkpoint-backed `SecMI` attack execution with real experiment results
- reproducible benchmark tables for black-box, white-box, and gray-box comparisons
- a finalized reporting layer for audit conclusions

## Repository Layout

```text
configs/                 audit, attack, dataset, and benchmark configs
docs/                    environment notes and project documents
experiments/             generated run artifacts and smoke outputs
notebooks/               exploratory notebooks
references/              project reference index and mirrored source materials
scripts/                 utility scripts and environment verification
src/diffaudit/           main project code
tests/                   unit and integration-oriented tests
third_party/secmi/       vendored minimal SecMI attack subset
external/                local exploratory clones only, ignored by git
workspaces/              collaboration work areas by research direction
```

## Research Workflow

Use this workflow for any new attack line:

1. select a target paper and define the attack assumption clearly
2. add or update a config file under `configs/`
3. implement or extend a planner / adapter layer
4. add a `dry-run` path before claiming real execution support
5. run smoke validation and record artifacts in `experiments/`
6. only then attempt checkpoint-backed experiments

For collaborative work, each contributor should also:

1. claim a workspace
2. add or update workspace notes before major code changes
3. keep shared code changes small and test-backed
4. record blockers explicitly when assets are missing

For `SecMI`, the recommended command order is:

1. `plan-secmi`
2. `prepare-secmi`
3. `dry-run-secmi`
4. real execution once assets exist

## Environment Setup

Create and activate the dedicated conda environment:

```powershell
conda env create -f environment.yml
conda activate diffaudit-research
python -m ipykernel install --user --name diffaudit-research --display-name "Python (diffaudit-research)"
```

This repository intentionally uses `conda` for the isolated Python runtime and `pip` for most scientific packages. That choice is documented in [docs/environment.md](docs/environment.md).

Verify the environment:

```powershell
$env:PYTHONPATH='src;.'
python scripts/verify_env.py
```

## Quick Start

Run the smoke pipeline:

```powershell
$env:PYTHONPATH='src;.'
python -m diffaudit run-smoke --config configs/benchmarks/secmi_smoke.yaml --workspace .
```

Build a `SecMI` plan from config:

```powershell
$env:PYTHONPATH='src;.'
python -m diffaudit plan-secmi --config configs/attacks/secmi_plan.yaml
```

Prepare the `SecMI` adapter context:

```powershell
$env:PYTHONPATH='src;.'
python -m diffaudit prepare-secmi --config configs/attacks/secmi_plan.yaml --repo-root third_party/secmi
```

Run a `SecMI` dry-run validation:

```powershell
$env:PYTHONPATH='src;.'
python -m diffaudit dry-run-secmi --config configs/attacks/secmi_plan.yaml --repo-root third_party/secmi
```

If required assets are missing, `dry-run-secmi` will report `blocked` and name the missing path instead of crashing.

## Reference Materials

The repository includes mirrored PDF materials under `references/materials/` for team study and reproducibility context. The index and categorization live in [references/README.md](references/README.md) and [references/materials/README.md](references/materials/README.md).

These files are tracked because the project currently depends on a shared reading set during the survey stage. Attribution to upstream papers is preserved in filenames and repository notes.

## Collaboration Rules

Use [AGENTS.md](AGENTS.md) for repository-level agent rules and [CONTRIBUTING.md](CONTRIBUTING.md) for human collaboration rules, branch discipline, workspace usage, and experiment naming conventions.

## Reproducibility Notes

Many attack paths depend on external assets that are not bundled with the repository, including:

- trained checkpoints
- `flagfile.txt` or equivalent training-time configuration dumps
- real dataset roots
- member or non-member split assets beyond the vendored smoke materials

Repository support should therefore be interpreted in stages:

- `code-ready`: planner, adapter, and tests exist
- `asset-ready`: required checkpoints and data paths exist locally
- `experiment-ready`: code and assets are both available

Do not treat a smoke run or dry-run as a benchmark result.

## Roadmap

Near-term repository goals:

- finish function-level `SecMI` execution integration
- add a real black-box benchmark configuration with asset contracts
- intake a white-box paper and a gray-box paper into the same adapter framework
- add a normalized result schema for experiment outputs and reports
- stabilize a clean multi-person collaboration workflow across the black-box, white-box, gray-box, and implementation workspaces

## Acknowledgements

This repository builds on public research code and papers, especially `SecMI`. Vendored files in `third_party/secmi/` retain attribution to the upstream project:

- [SecMI](https://github.com/jinhaoduan/SecMI)
