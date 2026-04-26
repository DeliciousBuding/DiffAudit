# 2026-04-18 X-101 04 H1 Risk-Targeted Unlearning Step-0 Prep

## Question

After `05` yielded the active slot to `04-defense`, can `04-H1 risk-targeted SISS / retain-forget mixture` move from family-selection wording into one real CPU-first prep surface on current admitted assets?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\src\diffaudit\defenses\risk_targeted_unlearning.py`
- `D:\Code\DiffAudit\Research\src\diffaudit\cli.py`
- `D:\Code\DiffAudit\Research\tests\test_risk_targeted_unlearning.py`
- `D:\Code\DiffAudit\Research\docs\report-bundles\gpt54\round2-results\04.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-packet-score-export-gsa-full-overlap-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-loss-score-export-targeted-full-overlap-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\cross-box\runs\crossbox-pairboard-gsa-targeted-full-overlap-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-prep-full-overlap-20260418-r1\summary.json`

## What Landed

### 1. One reusable Step-0 prep surface now exists in-repo

The repo now exposes `prepare-risk-targeted-unlearning-pilot`.

It does four bounded things:

1. load two score surfaces through the existing pairboard loader
2. align them on shared member/nonmember indices
3. orient scorer polarity and convert each split into within-split percentile ranks
4. export machine-readable forget/control lists for bounded `k` ladders

This is intentionally only a prep surface. It does not claim unlearning success and does not fake a defended verdict.

### 2. One real prep run now exists on current full-overlap assets

The first actual run used:

- `PIA exact packet export`
- targeted `GSA loss-score export`
- full-overlap shared board = `461 member / 474 nonmember`
- weights = `0.5 / 0.5`
- top fraction = `10%`
- ladders = `k = 16 / 32 / 64`

Artifacts now exist under:

- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-prep-full-overlap-20260418-r1\`

## Actual Read

### Shared-risk geometry

The main new fact is not “we have a list,” but **what the list geometry says**:

- `Top10%(GSA) ∩ Top10%(PIA)` member overlap = `8 / 461`
- so even `k = 16` cannot honestly stay inside pure intersection-only selection
- current `k = 16 / 32 / 64` ladders all therefore fall back to `aggregate-percentile`

This is useful because it hardens the next pilot contract:

- do not write the first `04-H1` run as “dual-consensus only”
- write it as `aggregate risk = 0.5 * q_gsa + 0.5 * q_pia`
- keep the top-fraction intersection count only as a diagnostic, not as the actual selector

### Exported ladders

Current mean combined-risk on exported sets:

- `k16`: forget `0.927535`, matched controls `0.902558`
- `k32`: forget `0.885202`, matched controls `0.870912`
- `k64`: forget `0.834387`, matched controls `0.822966`

Representative `k32` forget-set head:

- members: `6112, 9917, 5928, 1684, 4348, 2149, 4617, 2772`
- matched nonmembers: `3738, 1732, 3511, 4061, 3423, 7008, 7487, 3533`

## Verdict

- `x101_04_h1_step0_prep_verdict = positive but bounded`

More precise reading:

1. `04-H1` is now a real selected lane, not just a report-level default
2. the first honest next move is no longer family reselection
3. the next honest move is one actual bounded retain+forget pilot using one of the exported ladders
4. the current run still does **not** tell us whether unlearning reduces low-FPR leakage rather than just hiding it

## Next Contract

The next bounded pilot should stay narrow:

- default rung = `k32`
- selector = exported `aggregate-percentile` list
- objective = `retain + forget` hybrid (`L_keep` vs `L_keep - alpha * L_forget`)
- mandatory review = forgotten subset board + retained subset board + full split board
- mandatory later gate = defense-aware retrained attacker, not threshold-only transfer

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = 04-H1 actual retain+forget pilot on k32`
- `04 current state = Step-0 prep landed`
- `H2 adapter = fallback only`

## Canonical Evidence Anchor

Primary anchor:

- `D:\Code\DiffAudit\Research\workspaces\defense\runs\risk-targeted-unlearning-prep-full-overlap-20260418-r1\summary.json`

Supporting anchors:

- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-packet-score-export-gsa-full-overlap-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-loss-score-export-targeted-full-overlap-20260418-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\cross-box\runs\crossbox-pairboard-gsa-targeted-full-overlap-20260418-r1\summary.json`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Research/docs/comprehensive-progress.md`: update required
- `Research/docs/mainline-narrative.md`: update required
- `Research/docs/future-phase-e-intake.md`: update required
- `Platform/Runtime`: no direct handoff yet
