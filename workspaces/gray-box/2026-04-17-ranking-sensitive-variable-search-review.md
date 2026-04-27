# 2026-04-17 Ranking-Sensitive Variable Search Review

## Question

After black-box, gray-box second-defense, and white-box distinct-family import reviews all closed, which ranking-sensitive variable search is now the honest next gray-box CPU-first lane: reopening `PIA vs SecMI` disagreement, or pivoting to a more bounded `PIA vs TMIA-DM` switching/gating hypothesis?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/README.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-15-graybox-ranking-sensitive-disagreement-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-15-pia-vs-secmi-graybox-comparison.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-pia-vs-tmiadm-long-window-comparison.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-pia-vs-tmiadm-disagreement-exploitation-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`

## Review

### 1. `PIA vs SecMI` already answered the naive-disagreement question

Current `PIA vs SecMI` truth is:

- disagreement exists
- but rank correlation is still high enough that naive fusion does not beat the better single method
- the branch was explicitly left closed unless a new bounded hypothesis appears

That means reopening it without a concrete new variable would just restate the old negative verdict.

### 2. `PIA vs TMIA-DM` is the only branch that already shows bounded-positive ranking gain

Current `PIA vs TMIA-DM` truth is stronger:

- same-split rank correlation is only moderate on undefended aligned comparisons
- defended rank correlation is even lower
- trivial `z-score sum` already beats the best single method on both aligned undefended checks
- even on defended `GPU256`, the bounded combination still slightly beats defended `PIA`

So this pair already exposes a real ranking-sensitive opportunity rather than just an interpretive disagreement note.

### 3. The next honest variable search should be gating / switching, not another naive fusion rerun

The repo does not need another `sum` or `max` rerun.

What it does need is one bounded hypothesis that could explain when the pair helps:

- `confidence-gated switching`
- or another explicit rank-aware handoff rule on the same aligned surfaces

That is materially different from:

- naive `PIA + SecMI` averaging
- more same-family `TMIA-DM` scale churn
- another vague disagreement note without a decision rule

## Verdict

- `ranking_sensitive_variable_search_verdict = positive but bounded`
- the next honest gray-box CPU-first lane should be:
  - `PIA vs TMIA-DM confidence-gated switching design review`
- do **not** reopen `PIA vs SecMI` disagreement without a sharper hypothesis
- keep:
  - `gpu_release = none`
  - `next_gpu_candidate = none`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/implementation/challenger-queue.md`: update required
- `docs/comprehensive-progress.md`: light sync required
- `README.md`: light sync suggested
- `Platform/Runtime`: no direct handoff required
- `Leader/materials`: no immediate sync; this is still below headline replacement
