# 2026-04-17 X-21 Non-Graybox Next-Lane Reselection After X-20 Sync

## Question

After `X-20` completed the higher-layer stale-entry sync, what should now become the next honest non-graybox `CPU-first` lane?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`
- `<DIFFAUDIT_ROOT>/Research/docs/leader-research-ready-summary.md`
- `<DIFFAUDIT_ROOT>/Research/docs/competition-evidence-pack.md`
- `<DIFFAUDIT_ROOT>/Research/docs/competition-innovation-summary.md`
- `<DIFFAUDIT_ROOT>/Research/docs/competition-defense-qa.md`
- `<DIFFAUDIT_ROOT>/Research/docs/research-boundary-card.md`
- `<DIFFAUDIT_ROOT>/Research/docs/admitted-results-summary.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-x20-crossbox-system-stale-entry-sync-verdict.md`

## Candidate Comparison

### 1. Reopen `XB-CH-2` transfer / portability immediately

Not selected.

Reason:

1. the branch is still explicitly `needs-assets`;
2. `X-20` changed wording alignment, not the missing paired contracts;
3. another immediate reopen would just restate the same blocker.

### 2. Reopen `GB-CH-2` ranking-sensitive switching follow-up

Not selected.

Reason:

1. the first packet is already `negative but useful`;
2. no genuinely new gating variable has appeared;
3. reopening it now would be same-family churn rather than a stronger question.

### 3. Return to `I-A` as the next executable non-graybox lane

Selected.

Reason:

1. among currently executable non-graybox tasks, `I-A` remains the strongest near-term innovation track;
2. the recent sync passes reduced stale wording debt enough that a bounded residue audit is now the cleanest honest follow-up;
3. this keeps momentum on the strongest mechanism-backed line without pretending blocked branches are ready.

## Selection

- `selected_next_live_lane = X-22 I-A higher-layer truth-hardening residue audit after X-21 reselection`

## Verdict

- `x21_next_lane_reselection_verdict = positive`

More precise reading:

1. there is still no honest executable reopen in `XB-CH-2` or other queued blocked branches;
2. the strongest immediate non-graybox `CPU-first` move is to return to `I-A` for a bounded residue audit;
3. `next_gpu_candidate` stays `none`.

## Frozen Posture

- `active_gpu_question = none`
- `next_gpu_candidate = none`
- `current_execution_lane = X-22 I-A higher-layer truth-hardening residue audit after X-21 reselection`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance maintenance`

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-x21-next-lane-reselection-after-x20-sync.md`

## Handoff Decision

- `<DIFFAUDIT_ROOT>/ROADMAP.md`: update required
- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: update required
- `workspaces/implementation/challenger-queue.md`: update required
- `root_sync = required`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this step changes current control-plane ordering, but it does not change any admitted metric table or consumer schema.
