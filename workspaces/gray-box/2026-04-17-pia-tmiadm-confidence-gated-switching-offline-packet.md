# 2026-04-17 PIA vs TMIA-DM Confidence-Gated Switching Offline Packet

## Question

After the bounded-positive `PIA vs TMIA-DM` same-split evidence and the follow-up switching design review both landed, does the first real offline packet actually justify a stronger gray-box ranking-sensitive branch, or does it stay below headline / release / GPU escalation?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-17-pia-tmiadm-confidence-gated-switching-design-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-tmiadm-confidence-switch-20260417-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-tmiadm-confidence-switch-20260417-r1/analysis.md`
- `<DIFFAUDIT_ROOT>/Research/scripts/run_pia_tmiadm_confidence_switch.py`

## Packet Scope

This packet stays intentionally bounded:

- no new GPU run
- no new fitted classifier
- no label-aware gating
- no class-conditional routing
- only aligned attack-side scores already available on the current packet surfaces

The tested rule family is:

- normalize `PIA` and `TMIA-DM` scores
- compute dominant-method identity plus absolute margin gap
- trust the dominant method only when the frozen gap exceeds a threshold
- otherwise fall back to `z-score sum`

## Key Results

### Undefended aligned surfaces

`gpu128_undefended`:

- `PIA AUC = 0.817444`
- `TMIA-DM AUC = 0.836975`
- `z-score sum AUC = 0.868835`
- best gated threshold = `1.5`
- best gated AUC = `0.859863`

`gpu256_undefended`:

- `PIA AUC = 0.841293`
- `TMIA-DM AUC = 0.837814`
- `z-score sum AUC = 0.872986`
- best gated threshold = `1.5`
- best gated AUC = `0.864578`

Interpretation:

- on both undefended aligned packets, the confidence-gated rule does improve over simple dominant-only switching
- but it still does **not** beat the bounded `z-score sum` baseline

### Defended aligned surface

`gpu256_defended`:

- `PIA AUC = 0.829010`
- `TMIA-DM AUC = 0.717308`
- `z-score sum AUC = 0.832870`
- best gated threshold on this surface = `1.0`
- best gated AUC = `0.817078`

Interpretation:

- on the defended surface, confidence-gated switching underperforms both defended `PIA` and defended `z-score sum`
- so this packet does not justify a stronger defended gray-box headline

## Verdict

- `pia_tmiadm_confidence_gated_switching_offline_packet_verdict = negative but useful`

More precise reading:

1. the branch is still real and bounded, because the packet executes honestly and produces interpretable results;
2. but the current switching rule does **not** beat the already-landed `z-score sum` bounded combination on the two undefended aligned surfaces;
3. and it weakens further on the defended aligned surface;
4. therefore this lane should remain a useful ranking-sensitive analysis packet, not a promoted scorer family.

## Project-Level Interpretation

- this result does **not** change the gray-box headline:
  - `PIA + stochastic-dropout(all_steps)` remains the admitted defended mainline
- this result does **not** replace the current strongest defended challenger reference:
  - `TMIA-DM late-window + temporal-striding(stride=2)` remains the strongest defended challenger-specific branch
- this result does strengthen one honest sentence:
  - `PIA vs TMIA-DM` still exposes a real bounded ranking-sensitive relationship
  - but the first confidence-gated switching rule does not yet justify promotion beyond internal packet status

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `workspaces/implementation/challenger-queue.md`: update required
- `docs/comprehensive-progress.md`: optional light sync
- `Platform/Runtime`: no direct handoff
- `Leader/materials`: no immediate sync

## Carry-Forward Rule

- keep `gpu_release = none`
- keep `next_gpu_candidate = none`
- do not reopen this exact packet unless there is a genuinely new switching variable, not just another threshold scan
- the next honest move should be lane reselection, not more same-family packet inflation
