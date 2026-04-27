# 2026-04-17 Post-Gray-Box-Yield Next-Lane Reselection Review

## Question

After `GB-67` yielded the immediate slot, what should become the next live non-gray-box `CPU-first` lane, and what should stay frozen for `next_gpu_candidate` and the CPU sidecar?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-crossbox-closure-round-system-sync-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/2026-04-16-blackbox-next-lane-score-package-selection-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-distinct-whitebox-defended-family-import-selection-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-17-graybox-post-switch-lane-reselection-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-08-unified-attack-defense-table.md`

## Candidate Comparison

### 1. `I-A` truth-hardening

- `story impact`: highest
  - this is the nearest real innovation track that can still change project-level wording before any new run
  - it closes the exact gap already called out in root and research roadmaps: `formal statement + adaptive boundary + low-FPR`
- `asset readiness`: ready now
  - all required evidence already exists in admitted `PIA` baseline/defense artifacts plus adaptive-review notes
- `bounded cost`: CPU-only
  - no new run, no new asset dependency, no GPU admission needed

### 2. Black-box reopen or package follow-up

- `story impact`: lower than `I-A` right now
  - current black-box refresh already closed `negative but clarifying`
  - the remaining honest actions are maintenance, blocked assets, or same-family hold
- `asset readiness`: mixed to weak
  - `variation` still needs assets
  - `semantic-aux` still lacks a genuinely new family hypothesis
- `bounded cost`: CPU-only, but low leverage

### 3. White-box distinct-family follow-up

- `story impact`: low for the current round
  - the distinct-family import review already closed `none selected`
- `asset readiness`: not ready
  - visible options collapse into same-family corroboration, bounded branch continuation, or observability hold
- `bounded cost`: CPU-only, but would mostly produce repeated negative restatement

### 4. Cross-box transfer / portability follow-up

- `story impact`: still real, but not yet immediate
- `asset readiness`: insufficient
  - current honest state is still `needs-assets`
- `bounded cost`: CPU-first review is possible, but weaker than finishing `I-A`

## Verdict

- `post_graybox_yield_next_lane_reselection_verdict = positive`

More precise reading:

1. the next live non-gray-box `CPU-first` lane should now be:
   - `I-A truth-hardening`
2. the current execution shape should be frozen as:
   - `current_execution_lane = I-A trajectory-consistency truth-hardening`
   - `next_gpu_candidate = none`
   - `cpu_sidecar = PIA provenance / higher-layer boundary sync`
3. black-box and white-box should keep their current near-term closure verdicts:
   - no immediate black-box family reopen
   - no immediate white-box defended-family reopen
4. no new GPU question should be released until a genuinely new bounded hypothesis survives this CPU-first round

## Why This Lane Wins

- it is the strongest remaining path that can still change how the repo honestly explains the gray-box defense result
- it uses already-landed evidence instead of pretending another family is ready
- it improves both innovation truth and higher-layer consumability without spending GPU

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: update required
- `docs/admitted-results-summary.md`: update suggested once low-FPR contract is frozen
- `workspaces/implementation/challenger-queue.md`: light sync required so queue readers do not think reselection is still pending
- `Platform/Runtime`: no direct schema handoff yet
- `Leader/materials`: suggestion-only after `I-A` wording lands
