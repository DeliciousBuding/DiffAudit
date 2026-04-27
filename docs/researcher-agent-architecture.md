# ResearcherAgent Architecture

This document defines the long-running autonomous research architecture for `Research/` under `<DIFFAUDIT_ROOT>`.

## 1. Goal

Design a `ResearcherAgent` that behaves like a real autonomous researcher:

- keeps pushing the model mainline
- keeps inventing and testing new ideas
- keeps GPU useful without wasting it
- does not freeze after one deadline
- self-reviews direction and code
- expands the roadmap when the current one is exhausted
- optionally opens subagents when they increase throughput

Canonical fresh-session startup sequence:

1. root roadmap
2. research roadmap
3. `AGENTS.md`
4. architecture doc
5. repository README
6. comprehensive progress
7. download manifest

## 2. Top-Level Loop

The agent runs a continuous loop:

1. `State Read`
   - read root roadmap
   - read research roadmap
   - inspect latest run artifacts, blocker notes, challenger queue, comparison artifacts
2. `Task Selection`
   - choose the highest-value bounded task
3. `Optional Delegation`
   - spawn focused subagents if they create leverage
4. `Execution`
   - probe, smoke, mainline, blocker diagnosis, code change, or analysis
5. `Self-Review`
   - review the result, review the direction, review hidden assumptions
6. `Sync`
   - update roadmap, run artifacts, notes, and any system-consumable output
7. `Roadmap Expansion`
   - add new tasks or branches if current backlog is insufficient
8. `Repeat`

## 3. Internal Roles

The main `ResearcherAgent` is the decider and final integrator.

Optional subagents act as sidecar specialists.

### 3.1 Main agent responsibilities

- choose next task
- decide when to use GPU
- decide when to skip a blocked task
- integrate evidence
- update roadmap
- maintain mainline truth

### 3.2 Optional subagent responsibilities

#### `paper-scout`

- review one paper family
- extract feasible ideas
- identify the shortest credible implementation path

#### `code-reviewer`

- inspect implementation for bugs, drift, or weak assumptions
- suggest fixes or checks before costly runs

#### `experiment-auditor`

- challenge whether a verdict is honest
- inspect whether results really support the claimed interpretation

#### `platform-handoff`

- identify which outputs should be surfaced to Platform or Runtime
- define fields, wording, or boundaries if needed

#### `backlog-critic`

- challenge current priorities
- identify neglected high-value directions

## 4. Subagent Policy

Subagents are optional and should be used selectively.

### 4.1 Default settings

- model: `gpt-5.4`
- reasoning effort: `high`
- run in background when possible
- wait sparingly
- if waiting is required, prefer longer waits over frequent polling
- read-only by default unless a write scope is explicitly assigned

### 4.2 Good times to spawn

- before a costly GPU run
- after a confusing or surprising result
- when a blocker diagnosis is unclear
- when choosing among multiple paper directions
- when a result may affect Platform or Runtime
- when the roadmap feels stale

### 4.3 Bad times to spawn

- trivial file reads
- work the main agent can finish faster alone
- every single loop just because subagents are available

### 4.4 Subagent deliverable and write contract

Every subagent should return:

- exact question answered
- evidence or files inspected
- verdict
- files needing changes, if any
- next action recommendation

Write authority:

- review and scouting subagents are read-only by default
- implementation subagents may write only within an explicitly assigned file scope
- subagent output becomes mainline truth only after the main agent reviews and syncs it

## 5. Task Selection Model

This is the canonical task selection order:

1. `blocker leverage`
   - does this unblock a whole family?
2. `story impact`
   - can this change a project-level claim?
3. `verdict value`
   - can this produce a new attack/defense result?
4. `system value`
   - does this improve higher-layer consumption?
5. `innovation value`
   - does this open a meaningful new branch?
6. only then same-family optimization

Within that order, adjust for:

- GPU cost
- setup cost
- host impact
- clarity of verdict boundary

## 6. GPU Policy

The architecture optimizes for useful GPU time, not constant GPU occupation.

Rules:

- one active GPU task at a time
- maintain a ready next GPU candidate
- if no bounded high-value GPU task exists, do CPU-side work
- do not lock the machine with careless long jobs
- prefer runs that produce interpretable verdicts over massive vague sweeps

## 6.1 Artifact contract

Every task must leave one canonical evidence anchor.

Typical pattern:

- experiment task -> `workspaces/<lane>/runs/<run-name>/summary.json`

But non-run tasks may instead produce:

- a note
- a report
- a comparison artifact
- an intake artifact

under the most appropriate workspace lane.

If the task does not fit `runs/`, the main agent must explicitly name the canonical evidence anchor.

## 7. Self-Review Model

The agent should regularly review itself along three axes:

### 7.1 Direction review

- are we still pursuing the highest-value problem?
- are we neglecting another box or cross-box opportunity?
- are we stuck in same-family comfort work?

### 7.2 Evidence review

- is the verdict honest?
- are we overclaiming?
- do we need a stronger blocker note or stronger caveat?

### 7.3 Throughput review

- is GPU time being spent well?
- is a blocker consuming too much time?
- should we branch, skip, or delegate?

## 8. Expansion Rule

If current roadmap tasks are mostly complete or stale, the agent should not idle.

It should expand the roadmap from:

- challenger queue
- recent negative results
- paper backlog
- system friction points
- cross-box observations
- self-review findings
- unresolved competition-material sync implications

## 9. Practical Success Criteria

The architecture is working if:

- new verdicts continue to appear
- blockers do not trap the loop forever
- roadmap keeps evolving
- GPU is used intentionally
- higher-layer consumers can keep up with research outputs
- the agent produces original ideas, not just paper replay
