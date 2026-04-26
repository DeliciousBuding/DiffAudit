# 2026-04-17 I-A Trajectory-Consistency Truth-Hardening

## Question

Can the current repository freeze a stronger, still-honest `I-A` statement for `PIA + stochastic-dropout`, so the innovation claim rests on a mechanism, a bounded adaptive-attacker reading, and mandatory low-FPR reporting rather than on `AUC` alone?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\docs\comprehensive-progress.md`
- `D:\Code\DiffAudit\Research\docs\admitted-results-summary.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-09-graybox-signal-axis-note.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-09-pia-gpu512-adaptive-ablation.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-10-pia-8gb-supporting-frontier-note.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-08-unified-attack-defense-table.md`
- `D:\Code\DiffAudit\Research\docs\competition-evidence-pack.md`

## Formal Mechanism Statement

Current repository-safe statement:

- `PIA` should be read as an attack on `epsilon-trajectory consistency`.
- On the current `DDPM/CIFAR-10` workspace-verified contract, member samples induce a more stable denoiser-response trajectory than non-members under the fixed `PIA` scoring path.
- `stochastic-dropout(all_steps)` is therefore not just a generic perturbation toggle; its honest role is to inject inference-time randomness into that denoiser trajectory, weakening the consistency signal that `PIA` exploits.

This is the strongest mechanism wording supported now because it is grounded in:

- the gray-box signal-axis note that already classifies `PIA` as an `epsilon-trajectory consistency` attack family
- the adaptive review showing baseline stability but nonzero defense-side score variance
- the paired baseline/defense metric drop on the same admitted contract

## Bounded Adaptive-Attacker Reading

Current honest adaptive-attacker boundary:

- supported:
  - repeated-query adaptive review with `query_repeats = 3`
  - mean aggregation over repeated-query scores
  - same admitted `PIA` contract and same defense setting
- not supported:
  - a fully defense-aware retrained attacker
  - an attacker that redesigns the scoring family after observing the defense
  - any claim that adaptive attackers are solved in general

What the current evidence actually says:

- baseline remains deterministic under the repeated-query review
- `all_steps` defense still lowers privacy metrics after the same bounded adaptive review
- on `GPU512 adaptive`:
  - baseline `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR = 0.841339 / 0.786133 / 0.058594 / 0.011719`
  - defended `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR = 0.828075 / 0.767578 / 0.052734 / 0.009766`

So the correct reading is:

- `I-A` is robust to one bounded repeated-query adaptive review
- `I-A` is not yet validated against a broader adaptive-attacker family

## Low-FPR Contract

For `I-A`, higher-layer reporting must now carry all four metrics together:

1. `AUC`
2. `ASR`
3. `TPR@1%FPR`
4. `TPR@0.1%FPR`

Current admitted `GPU512` pair deltas:

- `AUC`: `-0.013264`
- `ASR`: `-0.018555`
- `TPR@1%FPR`: `-0.005860`
- `TPR@0.1%FPR`: `-0.001953`

This contract matters because:

- `AUC` and `ASR` alone can hide whether the defense still leaks at strict operating points
- the current defended line does show low-FPR improvement, but the magnitude is much narrower than the headline `AUC/ASR` drop
- that narrower low-FPR gain is exactly why the result must stay `provisional`, not overstated as validated privacy protection

## Higher-Layer Wording Freeze

Allowed wording:

- `PIA exposes an epsilon-trajectory consistency signal on the current workspace-verified contract, and inference-time stochastic dropout weakens that signal under a bounded repeated-query adaptive review while preserving the admitted headline structure.`

- `The current defended result is mechanistic and provisional: it lowers AUC, ASR, and both low-FPR TPR metrics on the admitted GPU512 pair, but it remains bounded by checkpoint/source provenance and by the current adaptive-review scope.`

Not allowed:

- `privacy solved`
- `adaptive attacker defeated`
- `validated defense benchmark`
- `general diffusion defense claim beyond current DDPM/CIFAR-10 contract`

## Verdict

- `ia_trajectory_consistency_truth_hardening_verdict = positive but bounded`

More precise reading:

1. `I-A.1` is now satisfied by a formal mechanism statement that ties `PIA` to `epsilon-trajectory consistency`.
2. `I-A.2` is now satisfied at the current honest boundary by freezing the repeated-query adaptive review as the supported adaptive-attacker reading.
3. `I-A.3` is now satisfied by making `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR` mandatory in higher-layer reporting.
4. `I-A.4` is now satisfied by replacing generic “dropout defense” phrasing with mechanistic wording plus explicit anti-overclaim boundaries.

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: update required
- `docs/admitted-results-summary.md`: update required
- `Leader/materials`: suggested sync, because the innovation wording is now sharper
- `Platform/Runtime`: no direct schema change, but consumers should preserve the four-metric read order when reading admitted rows
