# 2026-04-16 CDI Paired-Scorer Summary-Layer Sync

## Question

After freezing the `CDI` paired scorer boundary, have the gray-box summary-layer entry points been updated so later sessions do not fall back to stale `SecMI-only` or `paired canary only` wording?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\workspaces\gray-box\README.md`
- `D:\Code\DiffAudit\Research\README.md`
- `D:\Code\DiffAudit\Research\docs\comprehensive-progress.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-cdi-paired-scorer-boundary-review.md`

## Sync Applied

### Gray-box README

- added `CDI` status guidance to the current-task entry list
- clarified that:
  - first canary remains `SecMI-only`
  - repaired `2048` paired surface is landed
  - `control-z-linear` is now the default internal paired scorer
  - this is still not a headline or external-evidence claim

### Research README

- removed stale wording that implied real-checkpoint `SecMI` execution was still unfinished
- added current gray-box/CDI progress wording:
  - `SecMI` full-split execution is already landed
  - `CDI` now has:
    - first internal canary
    - repaired paired `2048` surface
    - default internal paired scorer

### Comprehensive Progress

- added one concise gray-box bullet describing the new `CDI` state:
  - `collection-level audit extension`
  - repaired paired surface
  - default internal paired scorer
  - explicit non-headline / non-external boundary

## Verdict

- `cdi_paired_scorer_summary_sync_verdict = positive`
- summary-layer entry points now reflect the current `CDI` truth closely enough for handoff and fresh-session reuse
- this round does not change GPU posture
- this round does change materials/summary wording:
  - `competition_material_sync = recommended`

## Handoff Decision

- `Leader / materials`: recommended wording update
- recommended wording:
  - `CDI` is now a gray-box collection-level audit extension with a landed first internal canary
  - paired `PIA + SecMI` scoring is available as the default internal paired scorer on the repaired `2048` shared surface
  - it should not yet be described as headline scorer or external copyright-grade evidence
- `Platform/Runtime`: no schema change required
