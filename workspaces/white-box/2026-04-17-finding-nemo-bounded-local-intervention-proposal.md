# 2026-04-17 Finding NeMo Bounded Local Intervention Proposal

## Question

After freezing the minimum honest bridge and the first trusted localization observable, what is the first honest local intervention proposal that can be defined on current admitted white-box assets without pretending that neuron-level ablation, benchmark release, or paper-faithful `Finding NeMo` execution already exists?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-finding-nemo-minimum-honest-protocol-bridge.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-finding-nemo-bounded-localization-observable-selection.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-10-finding-nemo-mechanism-intake.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-10-finding-nemo-protocol-reconciliation.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-10-finding-nemo-observability-smoke-contract.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-10-finding-nemo-activation-only-canary-sketch.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\signal-access-matrix.md`

## Candidate Space Review

### 1. Hard ablation

Examples:

- zeroing a unit
- zeroing a full selector output

Why it loses for the first proposal:

- too easy to drift into neuron-level claim inflation
- too likely to become a quality-destructive intervention before any bounded contract exists
- too close to “mechanism proven, now ablate it,” which the repo does not yet support

### 2. Global perturbation disguised as localization

Examples:

- whole-layer noise injection
- whole-block dropout
- full-model activation scrambling

Why it loses:

- not local enough to count as `I-B`
- collapses back into generic perturbation rather than surgical defense
- does not leverage the newly frozen local observable

### 3. Fixed top-k channel attenuation mask

Candidate shape:

- one selector only
- one timestep only
- one small set of channels only
- multiplicative attenuation instead of hard ablation

Why it wins:

- local enough to stay within the currently honest activation bridge
- weaker and safer than hard ablation for a first proposal
- can be described without claiming neuron identity or mechanism proof

## Selected Intervention Proposal

The first bounded local intervention proposal is now frozen as:

- `single-selector top-k channel attenuation mask`

Exact proposal shape:

1. target surface
   - `mid_block.attentions.0.to_v`
2. target timestep
   - `999`
3. target object
   - channel indices on the exported activation tensor, not named neurons
4. selection rule
   - future bounded calibration should rank channels by absolute member/control activation delta under the already-frozen selector/timestep/sample-binding contract
5. intervention rule
   - multiply the selected channels by a fixed attenuation coefficient `alpha`
   - default proposal: `alpha = 0.5`
6. intervention size
   - bounded small mask only
   - default proposal: `top-k = 8`
7. execution scope
   - selector-local
   - timestep-local
   - no global model rewrite
   - no training
   - no `cross-attention` intervention

## Why This Proposal Is Honest

This proposal stays within current repo truth because it only assumes:

- the activation tensor exists
- the selector/timestep/sample binding is already review-ready
- the first observable is local channel activity, not a benchmark score

It does **not** assume:

- identified memorization neurons
- proven causal mechanism
- paper-faithful `Finding NeMo`
- released validation smoke

It is therefore the narrowest proposal that still deserves the name `local intervention`.

## Explicit Boundaries

Allowed reading:

- a future bounded intervention could attenuate a tiny set of locally selected channels on one fixed selector and one fixed timestep

Not allowed:

- `we already know the responsible neurons`
- `the intervention already protects privacy`
- `this is the second executable white-box defense family`
- `this is ready for GPU release`

## Verdict

- `finding_nemo_bounded_local_intervention_proposal_verdict = positive but bounded`

More precise reading:

1. `I-B.3` is now satisfied:
   - the first honest local intervention proposal is a `single-selector top-k channel attenuation mask`
2. this is still only a proposal:
   - no execution
   - no quality claim
   - no defense claim
3. this proposal is deliberately selector-local, timestep-local, and channel-local, so it stays distinct from generic perturbation or full ablation
4. the next honest question is now:
   - what quality-vs-defense contract would make this proposal reviewable?

## Next Step

- `next_live_cpu_first_lane = I-B.4 quality-vs-defense metric contract`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: light sync required
- `docs/reproduction-status.md`: light sync required
- `workspaces/white-box/plan.md`: update required
- `Leader/materials`: no immediate sync; still below release-facing wording
- `Platform/Runtime`: no direct handoff; do not treat this as a released defense control
