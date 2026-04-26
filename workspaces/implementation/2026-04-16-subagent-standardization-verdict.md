# 2026-04-16 Subagent Standardization Verdict

## Task

- `INF-3.4` record what should become standard and what should stay optional

## Question

- After bounded paper-scout, code-review, and backlog-critic trials, which subagent workflows should become standard parts of the autonomous research loop, and which should remain optional?

## Evidence Base

- `workspaces/implementation/2026-04-16-subagent-paper-scout-verdict.md`
- `workspaces/implementation/2026-04-16-subagent-code-review-verdict.md`
- `workspaces/implementation/2026-04-16-subagent-backlog-critic-verdict.md`

## Verdict

- `positive`

The current subagent program is useful enough to standardize selectively, but not broadly.

## What Should Become Standard

1. use a read-only paper-scout when several literature candidates exist and local execution distance differs sharply;
2. use a read-only code-review subagent for automation/tooling commits that can change audit truth or canonical evidence generation;
3. use a read-only backlog-critic after multiple branch closures or whenever the priority ladder may have drifted.

## What Should Stay Optional

1. opening subagents for trivial file reads or obvious single-path next steps;
2. opening subagents at the start of every loop by default;
3. any write-capable delegation unless a file scope is explicitly assigned and reviewed.

## Decision

Current decision:

- close `INF-3` current evidence round as `positive`
- keep the standard subagent set narrow, read-only, and trigger-based
- return main attention to model-line tasks once the workflow truth is synced

## Handoff Note

- `Platform`: no sync needed.
- `Runtime`: no sync needed.
- `Materials`: no sync needed.
