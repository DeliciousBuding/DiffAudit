# X-77 White-Box Bounded Loss-Score First Actual Packet After X-76 Evaluator Implementation

## Task

Execute the first honest bounded actual loss-score packet on top of the frozen `X-75/X-76` contract:

- `threshold-style`
- `shadow-oriented`
- `shadow-threshold-transfer`
- `extraction_max_samples = 64` per split

## Verdict

`positive but bounded`

The first real bounded actual packet now exists on current admitted `DDPM/CIFAR10` white-box assets, and it stays positive under the intended shadow-only transfer contract.

## Canonical evidence anchors

- export packet:
  - `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-loss-score-export-bounded-actual-20260418-r1\summary.json`
- threshold-eval packet:
  - `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-loss-score-threshold-eval-bounded-actual-20260418-r1\summary.json`

## Packet outcome

### Export

Command:

```powershell
conda run -n diffaudit-research python -m diffaudit export-gsa-loss-score-packet `
  --workspace D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-loss-score-export-bounded-actual-20260418-r1 `
  --repo-root D:\Code\DiffAudit\Research\workspaces\white-box\external\GSA `
  --assets-root D:\Code\DiffAudit\Research\workspaces\white-box\assets\gsa-cifar10-1k-3shadow-epoch300-rerun1 `
  --resolution 32 `
  --ddpm-num-steps 20 `
  --sampling-frequency 2 `
  --attack-method 1 `
  --prediction-type epsilon `
  --extraction-max-samples 64 `
  --device cpu
```

Result:

- `status = ready`
- target + 3 shadow pairs all exported successfully
- all eight score tensors landed with `sample_count = 64`

### Threshold evaluation

Command:

```powershell
conda run -n diffaudit-research python -m diffaudit evaluate-gsa-loss-score-packet `
  --workspace D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-loss-score-threshold-eval-bounded-actual-20260418-r1 `
  --packet-summary D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-loss-score-export-bounded-actual-20260418-r1\summary.json
```

Transferred target metrics:

- `AUC = 0.699463`
- `ASR = 0.632812`
- `TPR@1%FPR = 0.03125`
- `TPR@0.1%FPR = 0.03125`

Shadow-side frozen board:

- `score_direction = member-lower`
- `shadow AUC = 0.592801`
- `shadow ASR = 0.583333`

Diagnostic-only target self-board:

- `AUC = 0.699463`
- `ASR = 0.664062`
- same direction, but a different threshold

## Honest reading

This packet is useful because it is now a **real actual bounded packet** under the intended transfer contract, not merely a smoke or abstract evaluation rule.

But it is still below release-grade honesty:

- low-FPR values stay weak (`0.03125 / 0.03125`)
- the margin over chance is real but modest
- this is not enough to promote a new white-box headline or widen into `LiRA / Strong LiRA`

## Handoff

- `Platform`: no immediate handoff
- `Runtime-Server`: no immediate handoff
- competition/materials sync: note-level only; no headline promotion

## Next recommended lane

`X-78 white-box bounded loss-score post-first-actual-packet boundary review after X-77 actual packet`

That next lane should decide whether this branch now deserves:

- bounded follow-up packet(s),
- a stricter freeze as auxiliary bounded evidence,
- or clean reselection back to another non-graybox / innovation lane.
