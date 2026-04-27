# 2026-04-17 Finding NeMo Extraction-Side Bounded Cap Verdict

## Question

After `I-B.11` blocked the first admitted packet on execution-budget grounds, can the repository make the dual-run intervention review honestly bounded at the extraction stage rather than only at the evaluation stage?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-first-admitted-fixed-mask-packet-execution-budget-review.md`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/cli.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/gsa.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_gsa_adapter.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_gsa_observability_adapter.py`

## Implementation

The dual-run review surface now supports extraction-side bounding:

1. `run-gsa-runtime-intervention-review` accepts:
   - `--extraction-max-samples`
2. if `--extraction-max-samples` is omitted:
   - it falls back to `--max-samples`
3. `_extract_gsa_gradients_with_fixed_mask(...)` now truncates the enumerated dataset files before gradient extraction
4. runtime summary now records:
   - `runtime.extraction_max_samples`

This means the first bounded intervention review can now be bounded at both layers:

- extraction
- evaluation

## Verification

Test commands:

```powershell
conda run -n diffaudit-research python -m unittest <DIFFAUDIT_ROOT>/Research/tests/test_gsa_adapter.py
conda run -n diffaudit-research python -m unittest <DIFFAUDIT_ROOT>/Research/tests/test_gsa_adapter.py <DIFFAUDIT_ROOT>/Research/tests/test_gsa_observability_adapter.py
```

Result:

- `test_gsa_adapter.py`: `4` tests pass
- combined `gsa_adapter + gsa_observability_adapter`: `12` tests pass

The updated intervention-review test now verifies:

1. `runtime.extraction_max_samples`
2. fallback from `max_samples`
3. that the extractor receives the bounded cap

## What Landed

The repo no longer has an execution-budget contradiction between:

- bounded evaluation
- full-board extraction

Current honest reading:

1. the dual-run review surface is still below admitted result level;
2. but it is now a **truly bounded** review surface rather than only a bounded readout layer.

## Verdict

- `finding_nemo_extraction_side_bounded_cap_verdict = positive but bounded`

More precise reading:

1. `I-B.12` is now satisfied:
   - extraction-side boundedness is implemented on the target-anchored fixed-mask review surface
2. the honest reading is:
   - `execution-budget blocker cleared`
3. this still does not mean a defense result exists:
   - the actual admitted packet has not yet run

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-extraction-side-bounded-cap-verdict.md`

## Next Step

- `next_live_cpu_first_lane = I-B.13 launch review for first truly bounded admitted target-anchored fixed-mask intervention-on/off packet`
- `next_gpu_candidate = I-B.13 actual target-anchored fixed-mask intervention-on/off bounded packet on admitted GSA assets`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: update required
- `workspaces/implementation/challenger-queue.md`: update required
- `workspaces/white-box/plan.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
