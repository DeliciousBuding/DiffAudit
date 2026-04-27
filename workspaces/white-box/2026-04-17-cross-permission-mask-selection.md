# 2026-04-17 Cross-Permission Mask Selection

## Question

Given the `I-C.1` falsifiable packet, which exact internal units or masks are honest to test first on the current admitted observability surface, without inflating channel indices into neuron claims or widening the intervention beyond the already-frozen locality budget?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-cross-permission-falsifiable-minimal-experiment.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-bounded-localization-observable-selection.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-bounded-local-intervention-proposal.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/signal-access-matrix.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-10-finding-nemo-activation-export-adapter-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-16-whitebox-feature-trajectory-verdict.md`

## Selection Principle

The first `I-C` mask family must inherit the narrowest already-honest white-box object:

1. channel indices on the exported activation tensor
2. one selector only
3. one timestep only
4. one small locality budget only
5. no neuron naming
6. no cross-attention intervention claim

Anything wider belongs to a later rung, not to the first packet.

## Selected Internal Unit Type

The first test object is now frozen as:

- `channel indices` on the raw activation tensor

More precise boundary:

- selector:
  - `mid_block.attentions.0.to_v`
- timestep:
  - `999`
- object granularity:
  - channel positions on the exported tensor
- not allowed to be described as:
  - identified memorization neurons
  - cross-attention neurons
  - causal units

## Calibration Source

The first mask family is calibrated from one fixed white-box canary pair already admitted by the adapter review:

- member:
  - `target-member/00-data_batch_1-00965.png`
- control:
  - `target-nonmember/00-data_batch_1-00467.png`
- checkpoint:
  - `checkpoint-9600`

Selection statistic:

- per-channel absolute member/control activation delta on the fixed selector/timestep surface

Boundary:

- this calibration pair is allowed for mask construction
- it is not, by itself, enough to establish cross-permission support

## First Mask Family

The first bounded mask set is now frozen as three masks with the same locality budget.

### 1. Primary targeted mask

- name:
  - `top_abs_delta_k`
- ranking rule:
  - take the channels with the largest absolute member/control activation delta
- default budget:
  - `k = 8`
  - `alpha = 0.5`

This is the primary mask because it is the narrowest instantiation of the already-frozen `I-B.3` local intervention proposal.

### 2. Matched random-mask control

- name:
  - `random_k_seeded`
- rule:
  - choose `k = 8` channels by a fixed recorded seed
- budget:
  - same selector
  - same timestep
  - same `k`
  - same `alpha`

This is required so `I-C` cannot hide behind generic perturbation.

### 3. Anti-target control mask

- name:
  - `bottom_abs_delta_k`
- rule:
  - choose the channels with the smallest absolute member/control activation delta
- budget:
  - same selector
  - same timestep
  - same `k`
  - same `alpha`

This is useful because it tests whether any small local mask works, or whether the effect is concentrated where the current white-box observable says it should be.

## Explicit Exclusions

The following are now explicitly outside the first `I-C` mask packet:

1. neuron IDs or neuron naming
2. `cross-attention` masks
3. multi-selector masks
4. multi-timestep masks
5. `grad_norm`-derived masks
6. full-layer attenuation
7. any budget wider than the current `selector=1 / timestep=1 / k=8 / alpha=0.5` default without declaring a new rung

## Why This Selection Is Honest

This selection stays inside current repo truth because:

1. it uses the only currently trusted localization object:
   - raw activation tensor channels
2. it preserves the already-frozen local intervention budget
3. it introduces both a random control and an anti-target control
4. it does not pretend that channel ranking already equals mechanism proof

## Verdict

- `cross_permission_mask_selection_verdict = positive but bounded`

More precise reading:

1. `I-C.2` is now satisfied:
   - the repository has one exact first-rung mask family for the falsifiable packet
2. the line remains below theory support:
   - the masks are selected
   - the cross-surface metric bundle is not yet frozen
   - no execution is authorized
3. the main gain is anti-drift:
   - future `I-C` work must now test a concrete targeted mask, not a vague “local unit” story

## Next Step

- `next_live_cpu_first_lane = I-C.3 define which black-box / gray-box / white-box metrics must move together to count as support`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: light sync required
- `workspaces/white-box/plan.md`: update required
- `workspaces/implementation/challenger-queue.md`: light sync required
- `Leader/materials`: no immediate sync; still hypothesis-only
- `Platform/Runtime`: no direct handoff; do not consume channel masks as released controls
