# 2026-04-16 Priority Ladder Resync

## Task

- roadmap priority-ladder resync after recent branch closures

## Question

- Does the current `Near-Term Priority Ladder` still reflect the latest executable research truth, or is it now stale enough to misroute the next autonomous step?

## Reviewed Signals

- `GB-1` text now says the cheap perturbation-style frontier is effectively exhausted for now
- `GB-3` is completed for the current wave
- `WB-3.2` now closes as `none selected / not-requestable`
- `X-3`, `BB-3`, `X-4`, and `WB-2` are already completed
- `INF-2` has completed its first automation-health round but still keeps `INF-2.3` open

## Verdict

- `positive`

The ladder was stale enough to mislead the next autonomous selection:

1. it still placed `GB-1` at the top even though the current text says that frontier is effectively exhausted without a new bounded hypothesis;
2. it still used already completed items (`X-3`, `BB-3`, `X-4`, `WB-2`) as if they were live priorities;
3. it still kept `WB-3` in the open queue even though the current candidate set is now explicitly `not-requestable`.

## Resync Decision

Priority should now prefer executable open questions:

1. `BB-1` second-signal black-box expansion
2. `INF-2` research automation health
3. `WB-4` white-box feature/trajectory upgrade
4. `INF-3` subagent leverage experiments

Conditional / hold items should remain visible, but not top-ranked:

- `GB-1` second gray-box defense
  - reopen only if a genuinely new bounded defense hypothesis appears
- `WB-3` white-box defense breadth
  - keep frozen below execution until a new family exists

## Handoff Note

- `Platform`: no sync needed.
- `Runtime`: no sync needed.
- `Materials`: no sync needed.
