# 2026-04-16 Subagent Code-Review Verdict

## Task

- `INF-3.2` test code-review subagent workflow

## Question

- Can a read-only code-review subagent materially improve automation or evidence-quality commits, or does it mostly restate what the main agent already sees?

## Setup

- subagent type: read-only code reviewer
- model: `gpt-5.4`
- reasoning: `high`
- write scope: none
- GPU use: none
- review target:
  - commit `80b4415`
  - `src/diffaudit/research_automation_health.py`
  - `tests/test_research_automation_health.py`
  - related evidence wording in `ROADMAP.md` and `workspaces/implementation/2026-04-16-run-anchor-consistency-verdict.md`

## Result

The subagent surfaced real follow-up issues rather than generic commentary:

1. active `summary.json` anchors that were present but merely `untracked` were not counted as friction;
2. git path classification was written in a Windows-specific way, which weakens cross-platform audit reliability;
3. legacy exclusion was unit-tested at the classifier level but not yet covered by an end-to-end audit test.

Follow-up hardening landed immediately in the mainline:

- active `untracked` anchors now count as actionable friction
- git path status checks no longer rewrite separators in a Windows-only way
- new end-to-end tests cover active-untracked friction and legacy exclusion

## Verdict

- `positive`

This workflow created real leverage. The subagent did not merely restate the patch; it found two correctness gaps and one evidence-hardening gap that were worth fixing before treating the automation audit as hardened.

## Standardization Note

What should become standard:

1. use a read-only code-review subagent for automation or tooling commits that can change audit verdicts, canonical evidence generation, or cross-machine truth;
2. require concrete file/line findings, not a general confidence statement;
3. treat subagent review as advisory until the main agent verifies and lands any follow-up fix.

What should stay optional:

1. markdown-only syncs with no executable logic change;
2. trivial local notes that do not affect automation outputs;
3. any write-capable code-review delegation.

## Decision

Current decision:

- close `INF-3.2` as `positive`
- keep code-review subagents as a standard tool for automation/evidence commits
- keep them optional elsewhere

## Handoff Note

- `Platform`: no sync needed.
- `Runtime`: no sync needed.
- `Materials`: no sync needed.
