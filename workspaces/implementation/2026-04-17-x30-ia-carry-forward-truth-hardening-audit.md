# 2026-04-17 X-30 I-A Carry-Forward Truth-Hardening Audit

## Question

After `X-29` returned the main slot to `I-A`, is there any currently visible higher-layer residue left in the mechanistic / low-FPR / bounded-adaptive reading of `PIA + stochastic-dropout`, or has the carry-forward wording now stabilized?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/docs/mainline-narrative.md`
- `<DIFFAUDIT_ROOT>/Research/docs/leader-research-ready-summary.md`
- `<DIFFAUDIT_ROOT>/Research/docs/competition-evidence-pack.md`
- `<DIFFAUDIT_ROOT>/Research/docs/competition-innovation-summary.md`
- `<DIFFAUDIT_ROOT>/Research/docs/competition-defense-qa.md`
- `<DIFFAUDIT_ROOT>/Research/docs/research-boundary-card.md`
- `<DIFFAUDIT_ROOT>/Research/docs/admitted-results-summary.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-x29-next-lane-reselection-after-x28-shared-surface-freeze-review.md`

## Review

### 1. Mechanistic wording is now consistently present

Current higher-layer entry points already converge on the same mechanistic reading:

- `PIA` attack signal = `epsilon-trajectory consistency`
- `stochastic-dropout(all_steps)` = inference-time randomization that weakens that signal

### 2. Low-FPR and bounded-adaptive requirements are still visible

The key competition- and leader-facing docs now consistently preserve:

- `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR`
- bounded repeated-query adaptive reading
- `paper-aligned blocked by checkpoint/source provenance`

### 3. No new high-layer residue is visible right now

The current issue is no longer stale `I-A` wording.

So the honest reading is:

- `I-A` remains the strongest near-term innovation track
- but the current carry-forward maintenance has stabilized enough to step back into sidecar status

## Verdict

- `x30_ia_carry_forward_truth_hardening_audit = positive but stabilizing`

More precise reading:

1. no new visible higher-layer `I-A` residue was found;
2. the current mechanistic + low-FPR + bounded-adaptive wording is stable enough for now;
3. `I-A` should remain a sidecar again unless new evidence or drift appears.

## Frozen Posture

- `active_gpu_question = none`
- `next_gpu_candidate = none`
- `completed_task = X-30 I-A carry-forward truth-hardening audit after X-29 reselection`
- `next_live_cpu_first_lane = X-31 non-graybox next-lane reselection after X-30 I-A audit`
- `carry_forward_cpu_sidecar = cross-box / system-consumable wording maintenance`

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-x30-ia-carry-forward-truth-hardening-audit.md`

## Handoff Decision

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`: update required
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`: update required
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`: update required
- `<DIFFAUDIT_ROOT>/Research/docs/mainline-narrative.md`: update required
- `<DIFFAUDIT_ROOT>/ROADMAP.md`: update required
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this pass stabilizes wording truth only; it does not change admitted metrics, schemas, or runtime requirements.
