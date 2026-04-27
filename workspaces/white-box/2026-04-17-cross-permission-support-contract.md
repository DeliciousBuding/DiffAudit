# 2026-04-17 Cross-Permission Support Contract

## Question

After freezing the first falsifiable packet and the first exact mask family, which white-box, gray-box, and black-box metrics are actually allowed to count as cross-permission support, and which combinations must still read as insufficient, mismatched, or negative?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/docs/admitted-results-summary.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-08-unified-attack-defense-table.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-cross-permission-falsifiable-minimal-experiment.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-cross-permission-mask-selection.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-quality-vs-defense-metric-contract.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-ia-trajectory-consistency-truth-hardening.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/2026-04-10-recon-decision-package.md`

## Core Principle

`I-C` cannot be supported by one metric moving on one permission surface.

It also cannot be supported by casually mixing:

1. white-box local packet deltas
2. gray-box split-level attack metrics
3. black-box admitted headline metrics from a different semantic surface

So the contract must be tiered.

The correct question is not:

- `did some AUC drop somewhere?`

The correct question is:

- `did the targeted local intervention produce the predicted white->gray co-movement, beat same-budget controls, and avoid being explained away by drift or surface mismatch?`

## Metric Tiers

### Tier 0: White-box local movement only

This tier is required for any interpretation, but it is never enough by itself.

Mandatory white-box metrics:

1. `selected_channel_abs_delta`
   - absolute member/control activation contrast on the selected channels
2. `selected_delta_retention_ratio`
   - post-mask selected contrast divided by pre-mask selected contrast
3. `off_mask_drift`
   - one off-mask or non-selected activation drift reading, so the result is not explained as whole-surface collapse

Required direction:

- targeted mask must reduce `selected_channel_abs_delta`
- targeted mask must reduce `selected_delta_retention_ratio`
- targeted mask must do so more strongly than both:
  - `random_k_seeded`
  - `bottom_abs_delta_k`

If only this tier moves, the reading is:

- `white-box local consistency only`
- not `cross-permission support`

### Tier 1: White-gray bridge support

This is the first tier that is allowed to count as real support for the current executable hypothesis.

Mandatory gray-box packet-local metric:

1. `PIA score gap on the matched member/control packet`
   - member score minus control score on the same packet used by the white-box local intervention review

Mandatory gray-box split-level bundle:

1. `AUC`
2. `ASR`
3. `TPR@1%FPR`
4. `TPR@0.1%FPR`

Mandatory gray-box reading rule:

- at least one of `AUC` or `ASR` must weaken in the predicted direction
- at least one low-FPR metric must also weaken in the predicted direction
- the targeted mask must outperform both same-budget controls
- the reading must survive the already-frozen `I-B.4` control-surface drift budget

If Tier 0 and Tier 1 both pass, the reading is:

- `white-gray bridge support`

This is the maximum honest positive reading available on the current overlap surface.

It is still not full project-level unification proof.

### Tier 2: Black-box corroboration

Black-box does matter for the long claim, but it does not belong inside the first minimal support tier.

Current repo truth:

- admitted `recon` metrics prove black-box risk exists
- they do **not** yet live on the same local `DDPM/CIFAR10` overlap surface

Therefore current black-box admitted rows are:

- relevant for project narrative
- not valid as first-packet `I-C` support metrics

For black-box to count as `I-C` corroboration later, it must satisfy all of:

1. a frozen compatible target family or surrogate local bridge
2. one explicit member/control score-gap reading or aligned attack bundle
3. semantic compatibility with the packet used for the white-gray bridge
4. no reliance on a completely different asset semantics such as current `public-subset / proxy-shadow-member` `recon`

If such a future black-box contract lands and co-moves in the same direction, the reading can be upgraded from:

- `white-gray bridge support`

to:

- `cross-box corroborated support`

Until then, black-box contributes:

- `none` to first-rung support counting

## Explicit Non-Support Cases

The following must now read as insufficient or negative:

1. white-box local delta shrinks, but gray-box `PIA` packet-local gap does not weaken
2. gray-box `AUC` weakens, but both low-FPR metrics worsen or stay contradictory
3. targeted mask and same-budget controls are indistinguishable
4. only black-box admitted `recon` metrics look strong, while the local white-gray bridge is absent
5. apparent support comes only after exceeding the frozen locality budget
6. apparent support is accompanied by drift large enough to make generic degradation the better explanation

## Support Ladder

The ladder is now frozen as:

1. `no support`
   - no reliable white-box local movement, or targeted mask does not beat controls
2. `white-box local only`
   - Tier 0 passes, Tier 1 fails
3. `white-gray bridge support`
   - Tier 0 and Tier 1 both pass
4. `cross-box corroborated support`
   - Tier 0 and Tier 1 pass, and a future aligned black-box contract also co-moves

Current repository ceiling after this note:

- `white-gray bridge support` is the highest achievable honest reading for the next executable packet

## Why This Contract Matters

This contract prevents three common errors:

1. calling a white-box-only local effect “cross-permission”
2. calling an `AUC` drop without low-FPR support “unification evidence”
3. treating current admitted `recon` metrics as if they already aligned with the local white-gray packet

## Verdict

- `cross_permission_support_contract_verdict = positive but bounded`

More precise reading:

1. `I-C.3` is now satisfied:
   - the repository has an explicit support-counting contract for the first cross-permission packet
2. the current executable ceiling is explicit:
   - the next packet can at most prove `white-gray bridge support`
   - black-box remains future corroboration, not current first-rung evidence
3. this improves GPU governance:
   - a future packet can now be judged honestly instead of being promoted on one metric family alone

## Next Step

- `next_live_cpu_first_lane = I-C.4 bounded white-gray bridge packet release review`
- `next_gpu_candidate = bounded I-C white-gray targeted-mask packet on the local DDPM/CIFAR10 overlap surface`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: light sync required
- `workspaces/white-box/plan.md`: update required
- `workspaces/implementation/challenger-queue.md`: light sync required
- `Leader/materials`: no immediate sync; still below project-level unified-theory wording
- `Platform/Runtime`: no direct handoff; do not consume this as a validated cross-box alignment result
