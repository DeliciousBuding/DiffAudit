# 2026-04-17 Finding NeMo First Intervention-On/Off Bounded Review Contract Selection

## Question

After `I-B.6` landed one real local intervention packet and `I-B.8` landed one real bounded attack-side evaluation control, what is the first honest way to combine them into an intervention-on/off review contract without pretending the repository already has a defense result?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-finding-nemo-quality-vs-defense-metric-contract.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-finding-nemo-first-bounded-localization-intervention-packet-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-finding-nemo-bounded-attack-side-evaluation-packet-control-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\finding-nemo-first-bounded-localization-packet-20260417-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\finding-nemo-bounded-attack-side-eval-control-20260417-r1\summary.json`
- `D:\Code\DiffAudit\Research\src\diffaudit\cli.py`
- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\gsa_observability.py`
- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\gsa.py`

## Key Constraint

Current repo truth is split across two different surfaces:

1. `I-B.6`
   - real in-model intervention
   - but only on one fixed `member + control` packet
2. `I-B.8`
   - real bounded attack-side metric board
   - but only on reused admitted gradients
   - with no intervention applied during gradient extraction

Therefore the repository does **not** yet have an honest intervention-on/off attack-side review.

## Candidate Comparison

### 1. Treat `I-B.6` and `I-B.8` as if they already form one review

Why it loses:

- attack metrics and intervention drift live on different packet identities;
- `I-B.8` reuses baseline gradients and never sees an intervened extraction path;
- this would silently convert `execution-positive + control-positive` into fake defense evidence.

### 2. Re-select channels separately for target and every shadow checkpoint

Why it loses:

- the intervention object stops being one frozen local proposal;
- locality budget becomes model-dependent rather than fixed;
- any resulting improvement would be confounded by per-model retargeting.

### 3. Freeze one target-anchored intervention object and compare baseline vs intervened on the same bounded evaluation packet

Why it wins:

- preserves one honest intervention identity;
- keeps attack-side comparison on one shared bounded packet;
- allows the existing local canary drift proxy to remain attached as the mandatory no-sampling quality readout;
- stays on admitted `GSA` assets without subset duplication.

## Selected First Honest Contract

The first honest intervention-on/off bounded review contract is now frozen as:

- `target-anchored fixed-mask dual-run bounded attack-side review on admitted GSA assets`

More precise contract:

1. same admitted attack family:
   - `GSA epoch300 rerun1`
2. same bounded evaluation surface:
   - `max_samples = 64`
   - same target/shadow member and nonmember packet definition for both runs
3. same frozen intervention object for both target and shadow model families:
   - selector `mid_block.attentions.0.to_v`
   - timestep `999`
   - `mask_kind = top_abs_delta_k`
   - `k = 8`
   - `alpha = 0.5`
   - channel indices frozen from the already executed native packet:
     - `[374, 471, 269, 1, 62, 360, 187, 394]`
4. same checkpoint-side identity:
   - target remains anchored on `checkpoint-9600`
   - shadows use their corresponding admitted checkpoint roots
   - no per-model mask reselection is allowed in the first review contract
5. the first review must contain two coupled artifacts:
   - `baseline bounded attack-side board`
   - `intervened bounded attack-side board`
6. the first review must also carry the existing mandatory local drift packet:
   - the native `965 / 467` canary-control packet remains the quality/locality anchor
7. the verdict must always be read jointly:
   - attack metrics from the bounded board
   - no-sampling drift from the local packet
   - locality budget fields
   - compute-cost fields

## Explicit Anti-Overclaim Rules

The first review is **not** allowed to:

1. reuse precomputed baseline gradients as the intervened side;
2. re-select `top-k` channels separately per checkpoint or per split;
3. replace the canary/control drift packet with attack metrics alone;
4. upgrade to full-scale admitted board before the bounded contract is executed honestly;
5. read any metric movement as defense-positive unless the on/off comparison happens on the same bounded packet.

## Verdict

- `finding_nemo_first_intervention_on_off_bounded_review_contract_selection_verdict = positive`

More precise reading:

1. `I-B.9` is now satisfied:
   - the first honest intervention-on/off review contract is selected
2. the repository still does not yet have a defense review result:
   - this is a contract freeze, not an executed defense packet
3. the next honest task is implementation of that dual-run bounded review surface.

## Canonical Evidence Anchor

- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-finding-nemo-first-intervention-on-off-bounded-review-contract-selection.md`

## Next Step

- `next_live_cpu_first_lane = I-B.10 implement target-anchored fixed-mask intervention-on/off bounded attack-side review surface on admitted GSA assets`
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
