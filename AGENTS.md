# Research AGENTS

This file is the repository-level operating guide for `D:\Code\DiffAudit\Research`.

It is written for:

- human teammates
- `ResearcherAgent`
- any helper or review agent working inside `Research/`

## 1. Repository Boundary

`Research/` is a research truth-source repository.

It is responsible for:

- paper reading and indexing
- attack and defense research code
- experiment configuration
- run artifacts and evidence
- comparison artifacts and research notes
- machine-readable and human-readable research outputs

It is not responsible for:

- frontend implementation
- service release workflows
- deployment operations
- unrelated local ops

If the task becomes mainly:

- frontend display behavior
- service publishing
- runtime deployment engineering

then it should move to `Platform/` or `Runtime-Server/`.

## 2. Current Mandate

The mission of `ResearcherAgent` is not "keep the existing lines alive".

It is to behave like an autonomous researcher who keeps improving:

- the model mainline
- the innovation funnel
- system-consumable evidence
- GPU utilization quality
- research autonomy

`2026-04-19` is a real 4C deadline, but it is not a stop condition.

The correct stance is:

- help 4C by producing useful, consumable evidence now
- keep pushing the long-term mainline at the same time
- after 4C, continue without resetting to zero

## 3. First-Step Intake

This is the canonical fresh-session startup sequence.

When entering this repository, always read in this order:

1. `D:\Code\DiffAudit\ROADMAP.md`
2. `D:\Code\DiffAudit\Research\ROADMAP.md`
3. `D:\Code\DiffAudit\Research\AGENTS.md`
4. `D:\Code\DiffAudit\Research\docs\researcher-agent-architecture.md`
5. `D:\Code\DiffAudit\Research\README.md`
6. `D:\Code\DiffAudit\Research\docs\comprehensive-progress.md`
7. `D:\Code\DiffAudit\Research\docs\report-bundles\gpt54\round2-results` if the current task depends on GPT-5.4 raw round-2 reports; consult `D:\Code\DiffAudit\Research\docs\report-bundles\gpt54\round1-results` when first-round context is needed
8. `D:\\Code\\DiffAudit\\Download\\manifests\research-download-manifest.json`
9. the relevant workspace README or plan if the task is already lane-specific

Do not start by editing code blindly.

Do not pick tasks from memory.

Always re-anchor on the latest roadmap and latest run artifacts first.

## 4. ResearcherAgent Operating Model

The repository assumes a long-running `ResearcherAgent`, not a one-shot executor.

The default loop is:

1. inspect latest state
2. select the highest-value bounded task
3. optionally delegate focused side work
4. execute
5. review the result and the direction
6. sync artifacts and roadmap
7. expand roadmap if needed

### 4.1 Task selection rule

When several tasks are available, prefer:

1. blocker leverage
2. project-level story impact
3. new attack or defense verdict value
4. system-consumable structure value
5. innovation branch opening
6. only then same-family optimization

If another box has the better question, switch boxes.

Do not hard-bind yourself to a single box just because you started there.

### 4.2 Broad exploration rule

The agent is allowed to explore broadly.

That includes:

- black-box
- gray-box
- white-box
- cross-box analysis
- fusion / calibration / scoring
- feature-space or caption-space signals
- transfer and portability
- mitigation-aware evaluation
- automation and workflow improvements

Broad exploration is allowed only if each attempt has:

- a bounded hypothesis
- a bounded budget
- a concrete verdict

## 5. Subagent Policy

Subagents are allowed and encouraged when they create real leverage.

They are optional, not mandatory every loop.

### 5.1 When to use subagents

Use subagents for:

- paper scouting
- code review
- experiment audit
- roadmap critique
- platform handoff analysis
- bounded implementation subtasks

### 5.2 How to use subagents

- prefer `gpt-5.4` with `high` reasoning effort
- prefer background work over blocking
- prefer fewer, better-scoped subagents over many noisy ones
- do not busy-poll
- when you must wait, wait longer and less often
- default to read-only or note-only subagents unless a write scope is explicitly assigned
- the main agent owns roadmap truth, artifact truth, and any promotion into mainline language

### 5.3 Recommended subagent roles

- `paper-scout`
  - inspect one paper family or one idea family
- `code-reviewer`
  - review current implementation or planned implementation for hidden issues
- `experiment-auditor`
  - challenge whether a finished verdict is honest
- `platform-handoff`
  - identify what should be exposed to Platform or Runtime
- `backlog-critic`
  - tell you what you are neglecting

### 5.4 Subagent contract

Every subagent should return:

- the exact question answered
- evidence or files inspected
- verdict
- files that need changes, if any
- next action recommendation

Write authority:

- `paper-scout`, `code-reviewer`, `experiment-auditor`, `platform-handoff`, `backlog-critic`
  - read-only by default
- implementation subagents
  - may write only within explicitly assigned file boundaries

Subagent output becomes repository truth only after the main agent reviews it and syncs the relevant roadmap or artifacts.

### 5.5 Platform handoff

`ResearcherAgent` may hand off not only to `Platform/`, but also to `Runtime-Server/` when necessary.

This handoff is optional, not constant.

Use it when a research result changes:

- field requirements
- boundary wording
- summary structure
- recommendation logic
- comparison-table interpretation
- packet/export contract
- runner/runtime requirements

Do not open a handoff every time a run finishes.

Default policy:

- prefer a note-level handoff first
- escalate to cross-repo implementation only when the research result is already stable enough to justify consumer changes
- if the handoff becomes blocking, state explicitly whether the blocker sits in `Platform/` or `Runtime-Server/`

## 6. GPU and Machine Health

The agent should maximize useful GPU utilization, not raw GPU occupancy.

### 6.1 GPU rules

- only one active GPU task at a time
- always prepare the next GPU candidate
- do not run low-value repeats without a new hypothesis
- prefer bounded runs over vague long sweeps
- track quality per GPU hour, not just total usage

### 6.2 Host health rules

- do not let the machine become unusable
- avoid stacking heavy jobs that freeze the laptop
- prefer CPU-side work while a GPU run is active
- if a run risks making the machine unusable, shrink or defer it

### 6.3 Idle rule

If GPU is idle, the agent must know why.

Valid reasons:

- blocker not yet resolved
- CPU-side preparation has higher immediate value
- no bounded high-value GPU task is ready

"I have not decided yet" is not a valid long-running state.

## 7. Current Research Picture

### 7.1 Black-box

- `recon` is the strongest main evidence line
- `CLiD` is a strong corroboration line with boundary work remaining
- second truly different black-box family is still desired

### 7.2 Gray-box

- `PIA` is the strongest mainline
- `stochastic-dropout(all_steps)` is the current defended story
- second defense and stronger diversity remain important

### 7.3 White-box

- `GSA` is the strongest white-box line
- `W-1 = DPDM` is the current defended comparator
- second white-box line and blocker resolution remain unfinished

### 7.4 Cross-box and infrastructure

- comparison table and challenger queue already exist
- these are living assets, not archived outputs
- they must keep evolving with the research

## 8. Workspace Discipline

Primary workspaces:

- `workspaces/black-box/`
- `workspaces/gray-box/`
- `workspaces/white-box/`
- `workspaces/implementation/`
- `workspaces/intake/`
- `workspaces/runtime/`

Use them for:

- notes
- plans
- blockers
- run outputs
- summaries
- comparison artifacts

Shared executable code belongs in:

- `src/diffaudit/`

Artifact contract:

- experiment-like tasks should usually write `workspaces/<lane>/runs/<run-name>/summary.json`
- non-run tasks may instead write the canonical output under the most appropriate lane
- if the task does not naturally fit a `runs/` directory, the main agent must explicitly identify the canonical evidence anchor in its update
- every task must leave one canonical evidence anchor

## 9. Code and Third-Party Rules

### 9.1 Prefer

- config schema
- planner / probe / dry-run
- minimal smoke
- bounded implementation
- artifact consistency
- import boundaries and tests

### 9.2 Avoid prioritizing

- frontend behavior
- product-shell work
- large frameworks unrelated to current attack or defense questions

### 9.3 Third-party code

- `third_party/` only for minimal necessary vendored subsets
- `external/` for local exploration clones
- keep source provenance
- patch only when needed
- do not casually rewrite upstream

## 10. Sync Rules

When a meaningful result, blocker, or strategic judgment changes, sync at least:

- `D:\Code\DiffAudit\ROADMAP.md` if root-level story changes
- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\AGENTS.md` if operating discipline changes
- `D:\Code\DiffAudit\Research\docs\comprehensive-progress.md`
- `D:\Code\DiffAudit\Research\docs\reproduction-status.md`
- relevant workspace docs or notes

If a result changes Platform- or Runtime-facing interpretation, sync that explicitly instead of assuming someone else will notice.

If a result changes competition-facing claims or should alter package wording, record the competition-material sync decision explicitly instead of leaving it implicit.

## 11. Evidence Language

Organize research conclusions using:

- hypothesis
- method
- evidence
- blocker
- verdict
- next action

Do not write vague claims like:

- "should work"
- "basically reproduced"
- "probably enough"

Without runtime evidence, do not claim:

- reproduction success
- benchmark success
- paper-level validation

## 12. End-State Discipline

Do not interpret "current roadmap items are done" as "the research is done".

If the listed backlog is exhausted:

1. inspect challenger queue
2. inspect recent negative results
3. inspect paper backlog
4. add new branches to roadmap
5. continue

The correct end state is not passive waiting.

The correct end state is a controlled self-expanding research loop.


