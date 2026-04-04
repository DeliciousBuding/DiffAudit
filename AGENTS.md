# DiffAudit Repository Guide

This repository is a research codebase, not a generic product application.

## Mission

Prioritize:

1. reproducibility
2. clear experiment boundaries
3. minimal valid integrations
4. documentation and evidence
5. only then UI or workflow polish

## Research Focus

The default repository focus is `black-box membership inference on diffusion models`.

White-box and gray-box work are valid extensions, but they must not fork the repository into unrelated structures. Reuse the same config, adapter, dry-run, and result-recording patterns.

## Workspace Structure

The repository is organized for multiple collaborators.

Use these workspace areas intentionally:

- `workspaces/black-box/`
- `workspaces/white-box/`
- `workspaces/gray-box/`
- `workspaces/implementation/`

Workspace folders are for planning notes, reproduction tracking, reading summaries, and task ownership. Shared executable code belongs in `src/diffaudit/`.

## Coding Priorities

Prefer adding:

- config schemas
- adapter layers
- dry-run validation
- artifact and result recording
- import and execution smoke tests

Do not start with:

- complex frontends
- platform-style user management
- database-heavy orchestration
- broad “all attacks” frameworks with no working path

## Third-Party Code Policy

Use `third_party/` only for the smallest necessary vendored subset.

When vendoring third-party code:

- keep the vendored scope minimal
- add a local README with source attribution
- patch only what is needed for integration, portability, or testing
- avoid rewriting upstream logic unless there is a concrete compatibility reason

`external/` is for local exploratory clones only and must never be committed.

Do not put personal scratch files into shared source directories. If temporary files are needed, keep them under local ignored paths or clearly marked workspace scratch areas.

## Experiment Asset Policy

Checkpoints, training flag dumps, dataset roots, and member split files are experiment assets.

If assets are missing:

- prefer returning a `blocked` dry-run result
- do not claim the experiment is runnable
- do not fabricate outputs or benchmark conclusions

Always distinguish:

- `code-ready`
- `asset-ready`
- `experiment-ready`

## Research Integrity Rules

- Do not claim a paper has been reproduced without fresh execution evidence.
- Do not describe smoke outputs as benchmark results.
- Do not generalize black-box findings to white-box or gray-box settings.
- Do not present a single paper’s assumptions as universal facts.

## Testing Expectations

New repository behavior should generally come with tests first.

At minimum, maintain coverage for:

- config loading
- attack planning
- adapter preparation
- dry-run validation
- vendored module import smoke tests

## Documentation Expectations

When adding a new attack direction or major capability, update:

- `README.md`
- relevant example configs under `configs/`
- reference indexes under `references/`
- environment notes if dependencies changed

The repository should always answer:

- what already works
- what is blocked
- what assets are missing
- what command verifies the current claim

## Multi-Person Coordination

- Prefer one research direction per contributor at a time.
- Keep branch work focused on one of: `black-box`, `white-box`, `gray-box`, or `implementation`.
- Use small commits with explicit messages.
- Before changing shared interfaces, update the relevant workspace or repository docs.
- When a task is blocked by assets, record the blocker instead of leaving silent partial work.

## Public Repository Constraints

Before committing large files:

- check total size
- check largest file size
- keep mirrored research materials indexed

Prefer documenting why a large file is tracked.

## Output Style

Research conclusions should be expressed as:

- assumption
- method
- evidence
- blocker

Avoid vague status language such as “probably works” or “should be fine”.
