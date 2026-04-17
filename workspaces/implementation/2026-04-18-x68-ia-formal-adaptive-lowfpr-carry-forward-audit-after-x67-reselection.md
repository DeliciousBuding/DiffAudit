# 2026-04-18 X-68 I-A Formal/Adaptive/Low-FPR Carry-Forward Audit After X-67 Reselection

## Question

After `X-67` reselected the live slot back to `I-A`, does the current `I-A` contract still contain one real carry-forward task worth the main lane, or should `I-A` return to sidecar-only status again?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-17-ia-trajectory-consistency-truth-hardening.md`
- `D:\Code\DiffAudit\Research\docs\admitted-results-summary.md`
- `D:\Code\DiffAudit\Research\docs\competition-evidence-pack.md`
- `D:\Code\DiffAudit\Research\docs\competition-innovation-summary.md`
- `D:\Code\DiffAudit\Research\docs\leader-research-ready-summary.md`
- `D:\Code\DiffAudit\Research\docs\mainline-narrative.md`
- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\challenger-queue.md`

## Audit Result

The underlying `I-A` packet remains stable:

- mechanism: `PIA = epsilon-trajectory consistency`
- defense reading: `stochastic-dropout(all_steps) = inference-time randomization weakening that signal`
- adaptive boundary: `bounded repeated-query review only`
- reporting contract: `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR`

So the live question is no longer packet truth.

However, one real higher-layer residue still existed:

- `leader-research-ready-summary.md` still opened with a one-page headline table that only surfaced `AUC / ASR`
- the body under that table was already correct
- but the top-sheet read path still invited an `AUC/ASR-first` summary reading before the reader reached the bounded-adaptive and low-FPR caveats below

That means `I-A` still had one honest carry-forward task, but only a narrow one:

- repair the top-sheet read path
- do not reopen experiment work
- do not mint a new `I-A` hypothesis

## Verdict

- `x68_ia_formal_adaptive_lowfpr_carry_forward_audit_verdict = positive but stabilizing`

More precise reading:

1. `I-A` did still contain one live carry-forward task.
2. That task was `Leader` top-summary surface hardening, not mechanism hardening.
3. Once that one-page table stops silently collapsing gray-box reading to `AUC / ASR`, `I-A` should return to sidecar-only status again.

## Handoff Decision

- `Research/docs/leader-research-ready-summary.md`: update required
- `Research/ROADMAP.md`: update required
- `Research/docs/comprehensive-progress.md`: update required
- `Research/docs/mainline-narrative.md`: update required
- root `ROADMAP.md`: update required
- prompt/bootstrap docs: update required because current live lane should move after `X-68`
- `Platform/Runtime`: no schema change required
- competition/materials sync: yes, but wording-only
