# 2026-04-17 PIA Provenance / Higher-Layer Boundary Sync

## Question

After `I-A` truth-hardening landed, do the higher-layer `Research` entry docs now describe `PIA` with the same mechanistic wording, bounded adaptive-attacker boundary, low-FPR contract, and provenance blocker, or are they still drifting between partial readings?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\docs\competition-defense-qa.md`
- `D:\Code\DiffAudit\Research\docs\competition-innovation-summary.md`
- `D:\Code\DiffAudit\Research\docs\research-boundary-card.md`
- `D:\Code\DiffAudit\Research\docs\reproduction-status.md`
- `D:\Code\DiffAudit\Research\docs\admitted-results-summary.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-09-pia-provenance-dossier.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-10-pia-provenance-split-protocol-delta.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-17-ia-trajectory-consistency-truth-hardening.md`

## Review

### 1. Provenance blocker wording was mostly present, but the mechanism packet was not consistently carried upward

Before this sync round, the main higher-layer docs already remembered:

- `workspace-verified + adaptive-reviewed`
- `paper-aligned blocked by checkpoint/source provenance`

But several of them still compressed `PIA` down to:

- `GPU512 baseline + stochastic-dropout pairing`
- or a generic “adaptive-reviewed” story

without always carrying:

- `epsilon-trajectory consistency`
- the bounded repeated-query adaptive-attacker reading
- the mandatory `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR` read order

### 2. The right higher-layer boundary is now a two-part sentence, not one half

The current repository-safe gray-box sentence must preserve both halves:

1. mechanism half
   - `PIA` exposes `epsilon-trajectory consistency`
   - `stochastic-dropout(all_steps)` weakens that signal via inference-time randomization
2. boundary half
   - the current result is only `workspace-verified + adaptive-reviewed`
   - `paper-aligned` remains blocked by `checkpoint/source provenance`
   - the adaptive support is bounded to repeated-query review
   - low-FPR metrics must travel with `AUC / ASR`

If either half is dropped, higher-layer wording becomes misleading again.

### 3. The sync target is now complete for the main competition-facing entry points

This round updates the highest-value entry docs so that they all carry:

- mechanistic wording
- bounded adaptive wording
- low-FPR read order
- provenance blocker wording

without changing any admitted claim or GPU posture.

## Verdict

- `pia_provenance_higher_layer_boundary_sync_verdict = positive`

More precise reading:

1. the main competition-facing and summary-facing `PIA` entry docs are now aligned to the same mechanistic + provenance-safe wording;
2. this does **not** change the admitted gray-box result;
3. this does **not** reopen `paper-aligned` review;
4. this does remove a real summary-layer drift risk.

## Current Frozen Posture

- `current_task = PIA provenance / higher-layer boundary sync`
- `verdict = completed`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`
- `next_live_cpu_first_lane = I-B minimum honest protocol bridge`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: update required
- `docs/reproduction-status.md`: update required
- `Leader/materials`: suggested sync, because 4C talk-track wording is now tighter
- `Platform/Runtime`: no schema handoff required, but any consumer summary should preserve the four-metric read order for admitted gray-box rows
