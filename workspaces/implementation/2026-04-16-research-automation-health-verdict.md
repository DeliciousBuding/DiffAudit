# 2026-04-16 Research Automation Health Verdict

## Task

- `INF-2.1` identify friction points in current run/update workflow
- `INF-2.2` add bounded automation where it reduces repeated human babysitting

## Question

- What parts of the current autonomous research loop are still unnecessarily manual, and can one small audit tool reduce that repeated session setup cost without changing the research truth model?

## Executed Work

Added a bounded CPU-only audit utility:

- `src/diffaudit/research_automation_health.py`
- `scripts/audit_research_automation_health.py`
- `tests/test_research_automation_health.py`

Executed audit summary:

- `workspaces/implementation/artifacts/research-automation-health-20260416.json`

## Friction Points Identified

1. `next GPU candidate` markers exist, but they are scattered across markdown notes rather than surfaced in one quick health check.
2. `active_gpu_question` / GPU-idle reasoning also exists, but requires manual repo-wide searching.
3. canonical run anchors can point into `workspaces/*/runs/.../summary.json`, where `.gitignore` means the file may exist locally but still be silently untracked unless explicitly force-added.
4. near-term priority reading still costs more context than it should for a fresh autonomous loop.

## Added Automation

The new audit utility now reports, in one JSON:

- current `Near-Term Priority Ladder` sections from `ROADMAP.md`
- discovered `next GPU candidate` markers
- discovered GPU-state / idle-reason markers
- run-summary anchor checks with:
  - file existence
  - tracked / ignored / untracked git state

This does not replace judgment, but it removes repeated setup scanning and makes anchor hygiene visible.

## Verdict

- `positive`

The new automation itself is a positive addition, but the audited repository health is still `friction detected`, not “fully healthy”. The tool made two remaining friction classes explicit:

1. many canonical run summaries still live under ignored `workspaces/*/runs/` paths, so force-add discipline remains necessary;
2. some markdown/template locations still mention run-summary anchors that do not correspond to concrete tracked evidence.

That is still a useful outcome: the repeated-friction gap is now visible in one bounded audit instead of being rediscovered manually every session.

## Handoff Decision

- `Platform`: no sync needed.
- `Runtime`: no sync needed.
- `Materials`: no sync needed.
- `Research`: future sessions can use the audit output as a quick state read before opening a new expensive run.

## Next Recommendation

1. leave `INF-2.3` open specifically for template cleanup and anchor-consistency tightening
2. keep model-mainline selection separate from automation health; this tool is a sidecar, not a reason to stall attack/defense work
