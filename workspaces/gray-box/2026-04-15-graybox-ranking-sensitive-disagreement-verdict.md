# 2026-04-15 Gray-Box Ranking-Sensitive Disagreement Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `GB-2 / P2-GS-1`
- `track`: `gray-box`
- `canonical_run`: `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\secmi-pia-disagreement-20260415-r1\summary.json`
- `gpu_status`: `completed`

## Question

Can the current `PIA` and `SecMI` score disagreement expose a ranking-sensitive variable that is worth promoting into a gray-box fusion or gating branch?

## Evidence Snapshot

Compared on the same `CIFAR-10 1024 / 1024` split:

- `PIA AUC = 0.838630`
- `SecMI stat AUC = 0.884180`
- `simple average ensemble AUC = 0.868736`
- `Spearman correlation = 0.907588`
- `agreement_rate = 0.877441`
- `disagreement_rate = 0.122559`

Per-sample asymmetry exists, but it is modest and `SecMI`-leaning rather than jointly complementary:

- `member_secmi_only_correct = 77`
- `member_pia_only_correct = 36`
- `nonmember_secmi_only_correct = 84`
- `nonmember_pia_only_correct = 54`

## Verdict

Current verdict:

- `negative but useful`

Interpretation:

1. `PIA` and `SecMI` do not behave like two mostly independent rankers.
2. Their disagreement is real, but too small and too correlated to justify a naive fusion branch.
3. The cheapest current ensemble recipe does not beat the better single method.

Therefore this round closes `GB-2` as:

- `no promotion for simple fusion`
- `positive for understanding the signal landscape`

## What Changes In Mainline Truth

- `PIA` remains the defended gray-box mainline.
- `SecMI` should now be written as an independent corroboration line, not as a blocked placeholder.
- `PIA + SecMI` should not be promoted as a current gray-box ensemble story.

## Reopen Conditions

Only reopen this branch with a new bounded hypothesis such as:

1. class-conditional disagreement,
2. confidence-gated method switching,
3. a learned fusion that uses sample attributes rather than raw score averaging.

Do not reopen with another naive score average.

## Handoff Guidance

- `Platform / Runtime`: no mandatory schema change for this verdict; if higher layers expose gray-box method structure, prefer `mainline = PIA`, `corroboration = SecMI`, `fusion_verdict = no-go`.
- `competition materials`: update gray-box wording to say `SecMI` is now full-split local corroboration, while `PIA` remains the defended mainline and simple fusion is rejected.
