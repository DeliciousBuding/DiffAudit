# 2026-04-21 X-129 I-A Truth-Hardening After X-128

## Question

After `X-128` selected `I-A` as the next live non-graybox lane, is there still any material-facing residue that softens the current `I-A` claim below the intended `four-metric + bounded repeated-query adaptive + provenance-blocked` standard?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\docs\competition-evidence-pack.md`
- `D:\Code\DiffAudit\Research\docs\competition-innovation-summary.md`
- `D:\Code\DiffAudit\Research\docs\competition-defense-qa.md`
- `D:\Code\DiffAudit\Research\docs\leader-research-ready-summary.md`
- `D:\Code\DiffAudit\Research\docs\admitted-results-summary.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-21-x128-non-graybox-next-lane-reselection-after-x127.md`

## Findings

### 1. Core leader-facing surfaces were already mostly aligned

The strongest active `I-A` surfaces already preserved:

- four-metric read order
- `epsilon-trajectory consistency`
- `stochastic-dropout` as inference-time randomization
- provenance blocked boundary

### 2. The remaining residue was material-facing wording softness

Two competition-facing docs still used the looser phrase:

- `workspace-verified + adaptive-reviewed`

That wording is weaker than the current intended contract:

- `workspace-verified + bounded repeated-query adaptive-reviewed`

It does not break the research truth, but it weakens the attacker-model boundary in higher-layer materials.

### 3. The residue is now cleared

Updated:

- `docs/competition-evidence-pack.md`
- `docs/competition-innovation-summary.md`

Both now use the sharper `bounded repeated-query adaptive-reviewed` wording.

## Verdict

`positive`.

Sharper control truth:

1. the remaining material-facing `I-A` wording softness is now cleared
2. current `I-A` higher-layer statement again reads as:
   - four metrics together
   - bounded repeated-query adaptive-reviewed
   - provenance blocked
3. the next honest live lane becomes:
   - `X-130 non-graybox next-lane reselection after X-129 I-A hardening`

## Canonical Evidence Anchor

- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-21-x129-ia-truth-hardening-after-x128.md`

## Handoff

- `Research/ROADMAP.md`: yes
- `workspaces/implementation/challenger-queue.md`: yes
- `Platform/Runtime`: no

Reason:

This change sharpens research-side material wording only. It does not alter admitted metrics or system contracts.
