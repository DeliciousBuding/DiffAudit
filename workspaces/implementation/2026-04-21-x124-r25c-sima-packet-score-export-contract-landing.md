# 2026-04-21 X-124 R2-5c SimA Packet-Score Export Contract Landing

## Question

Can `SimA` now emit a pairboard-ready packet surface on the current DDPM/CIFAR10 line, so that `PIA + SimA` support-fusion / calibration review becomes honestly executable?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/sima_adapter.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/cli.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_sima_adapter.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/pia_adapter.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/crossbox_pairboard.py`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-21-x123-r25b-pia-sima-support-fusion-contract-review.md`

## Changes Landed

### 1. `SimA` now has a real packet export surface

Added:

- `export_sima_packet_scores(...)` in `src/diffaudit/attacks/sima_adapter.py`

The new surface emits:

- `scores.json`
- `sample_scores.jsonl`
- `member_scores`
- `nonmember_scores`
- `member_indices`
- `nonmember_indices`

and freezes the summary as:

- `track = gray-box`
- `method = sima`
- `mode = packet-score-export`
- `device = cpu`
- `gpu_release = none`
- `admitted_change = none`

### 2. CLI contract is now executable

Added:

- `export-sima-packet-scores`

in `src/diffaudit/cli.py`, aligned with the existing `PIA` packet-export surface.

### 3. Local regression coverage now protects both module and CLI paths

`tests/test_sima_adapter.py` now covers:

- direct `export_sima_packet_scores(...)`
- CLI `export-sima-packet-scores`

Both validate exact-index or packet-local export semantics and pairboard-ready score payloads.

## Verification

Passed:

- `python -m unittest <DIFFAUDIT_ROOT>/Research/tests/test_sima_adapter.py`

Observed unrelated existing failure outside this task:

- `python -m unittest <DIFFAUDIT_ROOT>/Research/tests/test_pia_adapter.py <DIFFAUDIT_ROOT>/Research/tests/test_sima_adapter.py`
- current unrelated drift is `run-pia-runtime-mainline --epsilon-precision-bins 32` not recognized by the present parser
- this task does not modify that parser surface and does not rely on it

## Verdict

`positive but bounded`.

Sharper control truth:

1. `SimA` packet-score export gap is now closed
2. `SimA` remains `execution-feasible but weak`; this is contract hardening, not challenger promotion
3. the next honest CPU-first lane is now:
   - `R2-5b PIA + SimA support-fusion / calibration review`

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-21-x124-r25c-sima-packet-score-export-contract-landing.md`

## Handoff

- `Research/ROADMAP.md`: yes
- `docs/comprehensive-progress.md`: yes
- `docs/research-autonomous-execution-prompt.md`: yes
- `docs/codex-roadmap-execution-prompt.md`: yes
- `workspaces/implementation/challenger-queue.md`: yes
- `Platform/Runtime`: no

Reason:

This change hardens a research-side packet contract and unlocks the next cross-surface gray-box review, but it does not change admitted metrics, Runtime endpoints, or Platform-facing schema.
