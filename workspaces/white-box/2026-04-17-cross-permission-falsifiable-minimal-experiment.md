# 2026-04-17 Cross-Permission Falsifiable Minimal Experiment

## Question

What is the smallest honest experiment that could falsify, rather than merely decorate, the `I-C` hypothesis that a bounded white-box local signal and the gray-box `PIA` signal are projections of the same memorization structure?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-ia-trajectory-consistency-truth-hardening.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-minimum-honest-protocol-bridge.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-bounded-localization-observable-selection.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-bounded-local-intervention-proposal.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-quality-vs-defense-metric-contract.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-16-whitebox-feature-trajectory-verdict.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-08-unified-attack-defense-table.md`

## Minimal Honest Surface

The first falsifiable packet must stay on the smallest overlap surface that is already defensible inside the repository:

1. one local `DDPM/CIFAR10` overlap surface only
2. one matched member/control sample packet that can be read on both the current white-box and gray-box paths
3. one bounded white-box local mask proposal imported from `I-B`
4. no black-box API surface
5. no conditional diffusion extrapolation
6. no training and no new model family

This is deliberately narrower than a full three-box theory test.

If the first packet cannot survive here, there is no honest reason to widen it.

## Experiment Shape

The first falsifiable experiment is now frozen as a `targeted-mask vs matched-random-mask directional co-movement test`.

It has four parts:

### 1. White-box targeted intervention

Apply one bounded selector-local, timestep-local attenuation mask derived from the already-frozen `I-B` observability surface.

Important boundary:

- `I-C.1` freezes the experiment shape, not the final channel list
- the exact unit or mask instantiation belongs to `I-C.2`

### 2. White-box internal readout

Measure whether the targeted intervention reduces the white-box member/control activation contrast on the same fixed selector/timestep packet.

The honest reading target is not “did some scalar move,” but:

- did the selected local activation contrast shrink in the predicted direction?

### 3. Gray-box external readout

On the same matched member/control packet, measure whether `PIA`'s member advantage also weakens in the same direction.

Current honest wording:

- this is a `trajectory-consistency` readout, not a black-box utility claim

### 4. Random-mask control

Compare the targeted intervention against one matched random mask with the same locality budget:

- same selector count
- same timestep count
- same `k`
- same `alpha`

Without this control, any drop could be explained by generic perturbation rather than structure-linked coupling.

## Directional Prediction

The `I-C` hypothesis survives this first packet only if a locally selected white-box mask produces directional co-movement that a matched random mask does not.

The predicted support pattern is:

1. targeted mask reduces white-box local activation contrast
2. targeted mask also reduces gray-box `PIA` member advantage on the same packet
3. this reduction is stronger than the matched random-mask control
4. the accompanying control-surface drift stays inside the already-frozen `I-B.4` review budget

## Falsifier

This first packet is explicitly falsified if any one of the following happens:

1. the targeted mask fails to reduce the white-box local activation contrast
2. the white-box local activation contrast shrinks, but gray-box `PIA` member advantage does not weaken on the same packet
3. the targeted mask and matched random mask produce indistinguishable gray-box weakening
4. the apparent gray-box weakening appears only together with excessive control-surface drift, so the effect is better explained as generic degradation

This is the key point:

`I-C` is not allowed to survive on “the story still sounds plausible.”

It must survive one packet where a white-box-local intervention predicts a direction on another permission surface better than a matched random perturbation.

## Why Black-box Is Not In The First Packet

The black-box surface is intentionally excluded from `I-C.1`.

Reason:

- adding black-box now would make the first packet larger, noisier, and easier to excuse
- the current honest first question is whether there is even a directional white-box -> gray-box coupling worth treating as a shared structure hypothesis

Black-box co-movement belongs to `I-C.3`, not to the first falsifier.

## Budget and Release Reading

This note does not authorize a run.

It freezes only the first falsifiable shape:

- one overlap surface
- one bounded local intervention
- one matched random-mask control
- one white-box internal readout
- one gray-box external readout

Current release posture remains:

- `active GPU question = none`
- `next_gpu_candidate = none`
- `gpu_release = none`

## Verdict

- `cross_permission_falsifiable_minimal_experiment_verdict = positive but bounded`

More precise reading:

1. `I-C.1` is now satisfied:
   - the repository has one explicit minimal falsifier for the cross-permission hypothesis
2. the hypothesis is still unverified:
   - nothing has been executed
   - black-box is not yet included
   - exact units or masks are not yet frozen
3. this note mainly improves honesty:
   - future `I-C` discussion must now survive a targeted-vs-random, cross-surface directional test

## Next Step

- `next_live_cpu_first_lane = I-C.2 define which internal units or masks would be tested`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: light sync required
- `workspaces/white-box/plan.md`: update required
- `workspaces/implementation/challenger-queue.md`: light sync required
- `Leader/materials`: no immediate sync; still hypothesis-only
- `Platform/Runtime`: no direct handoff; do not consume this as a validated unified signal claim
