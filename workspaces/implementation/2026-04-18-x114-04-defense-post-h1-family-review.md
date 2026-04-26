# 2026-04-18 X-114 04 Defense Post-H1 Family Review

## Question

After `X-113` closed the current `04-H1` scalar-tuning space as `negative but useful`, should `04-H2 privacy-aware adapter` now be promoted as the next honest candidate, or does `04` first need a broader CPU-side family review?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-18-x101-04-h1-risk-targeted-unlearning-step0-prep.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-18-x102-04-h1-first-actual-retain-forget-pilot.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-18-x106-04-h1-k16-changed-pilot-tri-board-review.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-18-x107-04-h1-k8-pure-intersection-followup-review.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-18-x108-04-h1-k16-next-followup-selection.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-18-x109-04-h1-k16-alpha-up-followup-review.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-18-x110-04-h1-post-alpha-parameter-selection-review.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-18-x112-04-h1-selective-variable-candidate-freeze.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-18-x113-04-h1-k16-mixture-lambda-down-followup-review.md`
- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\docs\comprehensive-progress.md`
- `D:\Code\DiffAudit\Research\docs\mainline-narrative.md`
- `D:\Code\DiffAudit\Research\src\diffaudit\defenses\risk_targeted_unlearning.py`
- `D:\Code\DiffAudit\Research\src\diffaudit\cli.py`

## What Was Checked

### 1. `04-H1` already has a real executable surface

The repo now has all of the following for `04-H1 risk-targeted SISS / retain-forget mixture`:

1. one CPU-first prep surface
2. one actual bounded pilot surface
3. one review surface
4. multiple real bounded follow-up runs

So `H1` is no longer just a family-selection sentence. It is an executed family with a real repo contract.

### 2. `04-H2 privacy-aware adapter` does not yet have the same repo status

The current repo search does **not** show:

1. one dedicated `04-H2` implementation module
2. one CLI entry for an adapter-style `04-H2` prep / run / review flow
3. one dedicated test surface for such a route
4. one real bounded `04-H2` packet or review board

The current `H2` truth is therefore still wording-level fallback, not execution-level fallback.

## Actual Read

### 1. Same-family `H1` scalar tuning is now honestly exhausted near-term

Current `04-H1` read is already narrow and stable:

1. `k32` is too weak
2. `k8` is cleaner but too weak
3. original `k16` is the best working instantiation
4. `k16 + alpha-up` failed
5. `k16 + mixture_lambda-down` failed

So the next honest move is no longer another same-family scalar rerun.

### 2. But that does not automatically promote `H2`

Promoting `H2 privacy-aware adapter` would require at least one honest executable surface:

1. minimal implementation contract
2. bounded CLI or script entry
3. basic test or dry-run proof
4. bounded asset contract

That surface does not yet exist in-repo.

So current repo truth is:

1. `H1` = executed family with bounded negative/mixed evidence
2. `H2` = adjacent fallback idea without an executable contract

This means `H2` cannot honestly become `next_gpu_candidate` directly.

### 3. `04` now needs family review, not immediate candidate promotion

The correct next move is one bounded CPU-first family review that answers:

1. is `H2 privacy-aware adapter` worth turning into a minimal executable surface at all
2. if yes, what is the smallest honest prep/probe contract
3. if no, should `04` temporarily hold after `H1` and let `05/02/03` reclaim the next experimental slot

## System-Consumption Read

No new `Runtime` or `Platform` schema is needed for this decision.

Current `Research -> Runtime -> Platform` read path already has the right shape:

1. admitted mainline tables stay unchanged
2. `04` control-plane wording just becomes sharper
3. if `H2` is later implemented, it should reuse existing admitted / candidate governance surfaces rather than invent a new protocol

## Verdict

- `x114_04_defense_post_h1_family_review_verdict = cpu-first family review before any H2 promotion`

More precise reading:

1. original `k16` remains the current best working instantiation
2. same-family scalar tuning is no longer an honest immediate GPU path
3. `H2 privacy-aware adapter` remains fallback wording only, not an execution-ready candidate
4. `04` therefore continues as a CPU-side family-review lane before any new GPU release

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `04 next move = bounded CPU-first family review`
- `Research -> Runtime -> Platform = contract-stable, no schema change required`

## Canonical Evidence Anchor

- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-18-x114-04-defense-post-h1-family-review.md`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Research/docs/comprehensive-progress.md`: update required
- `Research/docs/mainline-narrative.md`: update required
- `Platform/Runtime`: no code or schema update required for this review verdict
