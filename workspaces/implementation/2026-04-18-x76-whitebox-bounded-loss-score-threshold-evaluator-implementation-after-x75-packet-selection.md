# X-76 White-Box Bounded Loss-Score Threshold Evaluator Implementation After X-75 Packet Selection

## Task

Implement the first honest bounded threshold-style evaluator surface on top of `export-gsa-loss-score-packet`, without mutating admitted `run-gsa-runtime-mainline` semantics.

## Why this task

`X-74` already landed bounded internal loss-score export and `X-75` already froze the first honest packet contract:

- `threshold-style`
- `shadow-oriented`
- `shadow-threshold-transfer`
- `extraction_max_samples = 64` per split

The remaining blocker was no longer packet selection. It was evaluator implementation.

## Implementation verdict

`positive but bounded`

The repository now exposes a separate bounded evaluator surface:

- code surface: `evaluate_gsa_loss_score_packet(...)`
- CLI surface: `evaluate-gsa-loss-score-packet`

This evaluator:

- reads exported loss-score artifacts from `export-gsa-loss-score-packet`
- pools shadow member / nonmember scores
- freezes score orientation from pooled shadow scores only
- freezes the operating threshold from pooled shadow scores only
- transfers the frozen orientation + threshold onto target evaluation
- emits `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR`
- keeps a target self-board only as `diagnostic-only`

## Evidence

### Tests

- `conda run -n diffaudit-research python -m pytest D:\Code\DiffAudit\Research\tests\test_gsa_adapter.py`
  - `6 passed`
- `conda run -n diffaudit-research python -m unittest D:\Code\DiffAudit\Research\tests\test_gsa_adapter.py`
  - `OK`

### Real bounded smoke

Evaluator smoke command:

```powershell
conda run -n diffaudit-research python -m diffaudit evaluate-gsa-loss-score-packet `
  --workspace D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-loss-score-threshold-eval-bounded-smoke-20260418-r1 `
  --packet-summary D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-loss-score-export-bounded-smoke-20260418-r1\summary.json
```

Canonical run anchor:

- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-loss-score-threshold-eval-bounded-smoke-20260418-r1\summary.json`

Observed smoke truth:

- `status = ready`
- `mode = loss-score-threshold-eval`
- shadow pooled board inferred `score_direction = member-lower`
- frozen shadow threshold transferred cleanly onto target
- target transfer board was negative on the smoke packet:
  - `AUC = 0.0`
  - `ASR = 0.0`
  - `TPR@1%FPR = 0.0`
  - `TPR@0.1%FPR = 0.0`
- target self-board remained positive, which confirms why it must stay `diagnostic-only` rather than packet verdict

## Honest boundary

This does **not** yet create a release-grade loss-feature result.

Current smoke still reads only a tiny bounded packet (`extraction_max_samples = 1` on the smoke export source), and `X-75` already fixed that the first honest actual packet should be `64` per split. Low-FPR reporting is present, but current bounded evidence remains below release-grade honesty.

## Handoff

- `Platform`: no immediate handoff
- `Runtime-Server`: no immediate handoff
- competition/materials sync: note-level only; no claim upgrade

## Next recommended lane

`X-77 white-box bounded loss-score first actual packet after X-76 evaluator implementation`

That next lane should execute one honest bounded actual packet on the frozen contract rather than reopen `LiRA / Strong LiRA` or promote the current smoke beyond its honesty boundary.
