# 2026-04-17 X-63 I-A Formal/Adaptive Low-FPR Residue Audit After X-62 Reselection

## Status Panel

- `owner`: `ResearcherAgent`
- `task_type`: `cpu-first residue audit`
- `device`: `cpu`
- `verdict`: `positive`

## Question

After `X-62` returned the live lane to `I-A`, does any active higher-layer or materials-facing entry point still understate the current `formal + bounded adaptive + low-FPR` reading of `PIA + stochastic-dropout`?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-ia-trajectory-consistency-truth-hardening.md`
- `<DIFFAUDIT_ROOT>/Research/docs/admitted-results-summary.md`
- `<DIFFAUDIT_ROOT>/Research/docs/competition-evidence-pack.md`
- `<DIFFAUDIT_ROOT>/Research/docs/competition-innovation-summary.md`
- `<DIFFAUDIT_ROOT>/Research/docs/competition-defense-qa.md`
- `<DIFFAUDIT_ROOT>/Docs/competition-materials/prompts/2026-04-09-project-visual-prompt-pack.md`

## Review

### 1. Research-side high-level wording is already mostly aligned

The active research-side and competition-side summary docs already preserve:

- `PIA = epsilon-trajectory consistency`
- `stochastic-dropout(all_steps) = inference-time randomization`
- `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR`
- `bounded repeated-query adaptive review`

So the remaining residue is no longer in the main research truth-source docs.

### 2. One materials-facing visual prompt still lagged behind the current contract

The active visual prompt for the PIA attack-defense figure still said:

- `底部预留可放指标说明的位置，例如 baseline 与 defense 的 AUC 对比说明`

That wording is now stale because it invites a materials-layer collapse back to:

- `AUC-only defense telling`

which conflicts with the frozen `I-A` contract.

### 3. The honest fix is a materials-facing wording upgrade, not a new experiment

The prompt should explicitly preserve:

- four-metric reading
- bounded repeated-query adaptive boundary
- provisional / non-overclaim defense status

This is a true `I-A` residue cleanup because it affects how downstream materials may visualize or summarize the main gray-box story.

## Verdict

- `x63_ia_residue_audit_verdict = positive`
- one active materials-facing residue existed in the PIA visual prompt
- that residue is now fixed by replacing `AUC-only` guidance with:
  - `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR`
  - bounded repeated-query adaptive note
  - provisional/non-overclaim boundary
- `active_gpu_question = none`
- `next_gpu_candidate = none`

## Competition-Material Sync Decision

- `yes`
- the fix directly updates one competition-material prompt so materials generation no longer drifts back to `AUC-only` defense framing

## Next Lane

- `X-64 non-graybox next-lane reselection after X-63 I-A residue audit`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: update required
- `docs/mainline-narrative.md`: light update required
- root `ROADMAP.md`: update required
- `Docs/competition-materials`: updated in-place
- `Platform / Runtime`: no immediate handoff required
