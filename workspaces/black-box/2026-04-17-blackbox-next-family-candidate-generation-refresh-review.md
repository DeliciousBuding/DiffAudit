# 2026-04-17 Black-Box Next-Family Candidate-Generation Refresh Review

## Question

After the closure round froze current black-box wording around `Recon / semantic-auxiliary-classifier / CLiD / variation`, does black-box now contain any honest next-family candidate that should become the next live CPU-first lane?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\README.md`
- `D:\Code\DiffAudit\Research\docs\comprehensive-progress.md`
- `D:\Code\DiffAudit\Research\workspaces\black-box\plan.md`
- `D:\Code\DiffAudit\Research\workspaces\black-box\README.md`
- `D:\Code\DiffAudit\Research\workspaces\black-box\paper-matrix-2024-2026.md`
- `D:\Code\DiffAudit\Research\workspaces\black-box\experiment-entrypoints.md`
- `D:\Code\DiffAudit\Research\workspaces\black-box\2026-04-09-blackbox-method-boundary.md`
- `D:\Code\DiffAudit\Research\workspaces\black-box\2026-04-15-blackbox-second-signal-semantic-aux-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\black-box\2026-04-15-blackbox-new-family-semantic-auxiliary-classifier-feasibility.md`
- `D:\Code\DiffAudit\Research\workspaces\black-box\2026-04-15-clid-paper-alignment-audit.md`
- `D:\Code\DiffAudit\Research\workspaces\black-box\2026-04-16-post-second-signal-blackbox-next-question-review.md`
- `D:\Code\DiffAudit\Research\workspaces\black-box\2026-04-16-variation-asset-contract-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\challenger-queue.md`

## Review

### 1. Current black-box families are already frozen enough to expose the real gap

The repo truth after `BB-7` is already stable:

- `Recon` is the frozen black-box headline
- `semantic-auxiliary-classifier` is the leading new-family challenger
- `CLiD` is boundary-quality corroboration only
- `variation` is a formal secondary track with a contract-ready unblock path, but still asset-blocked

So the refresh question is no longer "which current branch should be rerun." It is only "does any candidate now deserve promotion as the next black-box family-selection lane."

### 2. The visible black-box candidates do not produce an honest immediate next-family promotion

Current candidate classes fail for different reasons:

1. `caption / semantic refresh`
   - this mostly collapses back into the already-landed `semantic-auxiliary-classifier` family
   - current scoring / fusion follow-ups were already reviewed as same-ordering refinements rather than new families

2. `variation`
   - this is not blocked by idea scarcity
   - it is blocked by missing real `query_image_root`, plus endpoint / budget / frozen-parameter completion
   - therefore it is an asset-unblock lane, not a candidate-generation win

3. `CLiD`
   - the remaining work is paper-alignment / evaluator compatibility / shadow-side asset truth
   - that is boundary tightening, not a new black-box family

4. `Kandinsky 10/10`
   - current repo state only preserves a diagnostics pocket for runtime slowness
   - there is no honest research-ready family contract attached to it yet

### 3. The only genuinely different family-shaped idea is currently better classified outside immediate black-box promotion

`dataset-audit-track` is real and valuable, but current repo truth already absorbed that family through `CDI` on the gray-box side:

- it is already a landed internal collection-level audit extension there
- re-promoting it as the next black-box family would duplicate existing repo truth
- that would blur taxonomy instead of increasing method diversity

So `dataset-audit-track` remains useful as a black-box-adjacent concept, but not as the next honest black-box live lane.

## Verdict

- `blackbox_next_family_candidate_generation_refresh_verdict = negative but clarifying`
- black-box currently still does **not** expose a ready next-family promotion candidate
- current black-box reopen conditions remain:
  - a genuinely new feature family, or
  - a real asset / boundary change
- current global posture should therefore remain:
  - `gpu_release = none`
  - `next_gpu_candidate = none`
- the next live CPU-first lane should move to:
  - `second gray-box defense mechanism selection`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/black-box/plan.md`: update required
- `workspaces/implementation/challenger-queue.md`: update required
- `docs/comprehensive-progress.md`: light sync required
- `README.md`: light sync suggested
- `Platform/Runtime`: no schema handoff required
- `Leader/materials`: suggest wording that black-box is now stable rather than currently innovation-leading
