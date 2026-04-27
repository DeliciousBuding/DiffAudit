# 2026-04-17 Finding NeMo Bounded Attack-Side Evaluation Packet Control Verdict

## Question

After `I-B.7` froze the first honest attack-side review surface to "one bounded evaluation-size override on admitted `GSA` assets", can the repository implement that control and produce one real bounded review packet without reopening GPU?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-bounded-attack-side-evaluation-packet-selection.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-first-bounded-localization-intervention-packet-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/runs/gsa-runtime-mainline-20260409-cifar10-1k-3shadow-epoch300-rerun1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/cli.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/gsa.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_gsa_adapter.py`

## Implementation

The bounded control is now implemented as:

1. `run-gsa-runtime-mainline --max-samples <n>`
2. one per-side evaluation cap applied before closed-loop preprocessing:
   - target member
   - target nonmember
   - each shadow member bundle
   - each shadow nonmember bundle
3. emitted runtime summary now records:
   - `runtime.max_samples`

This is intentionally an evaluation-surface control, not a new asset root and not a new GPU question.

## Verification

Test command:

```powershell
conda run -n diffaudit-research python -m unittest <DIFFAUDIT_ROOT>/Research/tests/test_gsa_adapter.py
```

Result:

- `3` tests pass
- new bounded control path is covered

## Executed Review Packet

CPU-only bounded review on admitted `GSA epoch300 rerun1` gradients:

- source attack family:
  - admitted `GSA`
- source summary:
  - `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/runs/gsa-runtime-mainline-20260409-cifar10-1k-3shadow-epoch300-rerun1/summary.json`
- bounded override:
  - `max_samples = 64`
- evaluation surface:
  - truncate member/nonmember gradient tensors per side before preprocessing
- GPU:
  - none

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/runs/finding-nemo-bounded-attack-side-eval-control-20260417-r1/summary.json`

## Result

The bounded attack-side evaluation control works on admitted `GSA` artifacts.

Bounded packet metrics:

- `auc = 0.988159`
- `asr = 0.90625`
- `tpr_at_1pct_fpr = 0.453125`
- `tpr_at_0_1pct_fpr = 0.0`
- `shadow_train_size = 89`
- `target_eval_size = 128`

Full-scale admitted reference:

- `auc = 0.998192`
- `asr = 0.9895`
- `tpr_at_1pct_fpr = 0.987`
- `tpr_at_0_1pct_fpr = 0.432`
- `shadow_train_size = 4200`
- `target_eval_size = 2000`

Interpretation:

1. the repository now has a real bounded attack-side evaluation-size control on the admitted `GSA` surface;
2. the first bounded review packet is much smaller than the admitted full-scale board while still preserving a strong attack signal;
3. low-FPR readings degrade sharply under the bounded packet, which is exactly why this control must be reported as a bounded review surface rather than a new headline board.

## Boundary

This is still **not** a defense-positive result.

What it proves:

- `I-B.8` implementation is real;
- admitted `GSA` attack-side review can now be shrunk honestly without duplicating assets;
- the bounded packet emits the required four attack metrics plus evaluation-size fields.

What it does **not** prove:

- intervention-on/off defense effect;
- quality-preserving privacy defense;
- neuron localization;
- any new GPU-worthy question;
- any reason to replace the admitted full-scale `GSA` headline.

## Verdict

- `finding_nemo_bounded_attack_side_evaluation_packet_control_verdict = positive but bounded`

More precise reading:

1. `I-B.8` is now satisfied:
   - bounded attack-side evaluation packet control exists on the admitted `GSA` surface
2. the honest reading is:
   - `control-positive / defense-unproven`
3. the next question is no longer whether bounded attack-side review is implementable:
   - it is how to join this bounded attack-side control with the existing local intervention packet into one first honest intervention-on/off review contract.

## Next Step

- `next_live_cpu_first_lane = I-B.9 select first honest intervention-on/off bounded attack-side review contract`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: update required
- `workspaces/implementation/challenger-queue.md`: update required
- `workspaces/white-box/plan.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
