# 2026-04-17 Finding NeMo Bounded Attack-Side Evaluation Packet Selection

## Question

After the first executable `I-B.6` packet landed as `execution-positive / defense-unproven`, what is the next honest bounded attack-side evaluation surface for the first quality-vs-defense review?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/plan.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-quality-vs-defense-metric-contract.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-first-bounded-localization-intervention-packet-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-08-whitebox-attack-defense-table.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/artifacts/unified-attack-defense-table.json`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/cli.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/gsa.py`

## Candidate Comparison

### 1. Reuse the admitted full-scale white-box board directly

Why it loses:

- too large for the first `I-B` review packet;
- collapses packet review into benchmark-like rerender;
- makes locality and quality drift harder to interpret.

### 2. Materialize a separate subset asset root first

Why it loses:

- adds avoidable asset drift;
- duplicates dataset semantics before the packet contract itself is stable;
- makes the first review depend on new asset management instead of one bounded evaluation control.

### 3. Add one bounded evaluation-size control on the admitted `GSA` runtime surface

Why it wins:

- stays on the admitted asset family;
- keeps the first attack-side review bounded;
- gives the packet one honest path to the four attack metrics required by `I-B.4`;
- avoids inventing a second asset contract just to make the first review smaller.

## Selected Next Surface

The next honest attack-side evaluation surface is now frozen as:

- `bounded local attack-side evaluation packet on the admitted GSA family`

More precise contract:

1. same admitted `GSA epoch300 rerun1` asset root
2. same attack family:
   - `GSA`
3. same white-box review packet semantics:
   - local in-model attenuation remains the intervention object
4. one new bounded evaluation control is required:
   - `max_samples` or equivalent explicit evaluation-size override on the target/shadow evaluation surface
5. no new asset root
6. no GPU release

## Verdict

- `finding_nemo_bounded_attack_side_evaluation_packet_selection_verdict = positive`

More precise reading:

1. `I-B.7` is now satisfied:
   - the next honest attack-side review surface is selected
2. the review should stay on admitted `GSA` assets with one bounded evaluation-size override
3. the next task is implementation, not more packet-shape debate.

## Next Step

- `next_live_cpu_first_lane = I-B.8 implement bounded attack-side evaluation packet control on admitted GSA surface`
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
