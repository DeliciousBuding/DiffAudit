# 2026-04-16 Run Anchor Consistency Verdict

## Task

- `INF-2.3` improve run artifact consistency or summary templates if needed

## Question

- Can the current automation-health audit distinguish real actionable anchor drift from expected template / legacy references, instead of mixing them into one noisy friction bucket?

## Executed Work

Updated the automation-health classifier:

- `src/diffaudit/research_automation_health.py`
- `tests/test_research_automation_health.py`

New behavior:

1. classify run-summary mentions as:
   - `template_placeholder`
   - `legacy_reference`
   - `active_reference`
2. compute actionable counts separately from total counts
3. treat active ignored / untracked / missing anchors as true friction

Updated audit artifact:

- `workspaces/implementation/artifacts/research-automation-health-20260416.json`

## Result

The refined audit now reports:

- `template_placeholder_run_summaries = 3`
- `actionable_run_summaries = 159`
- `actionable_tracked_run_summaries = 3`
- `actionable_ignored_run_summaries = 156`
- `actionable_untracked_run_summaries = 0`
- `actionable_missing_run_summaries = 2`

This is better than the previous mixed bucket because placeholder references are no longer conflated with active drift.

## Verdict

- `positive`

The repository still has real anchor hygiene friction, but the automation layer now reports it more honestly:

1. template placeholders are no longer treated as if they were active broken references;
2. the remaining friction signal is narrower and more actionable:
   - active ignored summaries
   - active untracked summaries
   - active missing summaries
3. this is enough to close the current `INF-2.3` round without pretending that all anchor drift is fixed.

## Decision

Current decision:

- close the current `INF-2.3` round as `positive`
- keep using the refined automation-health audit as the canonical way to detect active anchor drift
- do not hand-clean every legacy/template doc unless a live consumer actually needs it

## Handoff Note

- `Platform`: no sync needed.
- `Runtime`: no sync needed.
- `Materials`: no sync needed.
