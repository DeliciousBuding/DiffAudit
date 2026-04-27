# 2026-04-17 Cross-Permission Bridge Packet Release Review

## Question

Given the already-frozen `I-C.1 / I-C.2 / I-C.3` packet shape, mask family, and support-counting contract, is one bounded white-gray bridge packet now honest to release as the next active GPU question?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-cross-permission-falsifiable-minimal-experiment.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-cross-permission-mask-selection.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-cross-permission-support-contract.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-bounded-local-intervention-proposal.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-10-finding-nemo-activation-export-adapter-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/pia-intake-gate.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-07-pia-real-asset-probe.md`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/gsa_observability.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/pia_adapter.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/cli.py`

## Candidate Under Review

The candidate under review is:

- `bounded I-C white-gray targeted-mask packet on the local DDPM/CIFAR10 overlap surface`

Expected shape:

1. one white-box targeted mask
2. two same-budget controls
3. one white-box local activation readout
4. one gray-box matched-packet `PIA` score-gap readout
5. one bounded split-level gray-box metric bundle
6. one GPU run only if the packet is already executable and interpretable

## Review Axis 1: White-box Execution Surface

Current white-box truth is still:

- `read-only activation export adapter implemented`

What is available now:

1. fixed admitted checkpoint loading
2. fixed sample-pair activation export
3. fixed selector/timestep hook
4. raw tensor artifact export

What is **not** available now:

1. a mask-application executor on the admitted white-box surface
2. a post-mask activation recapture path
3. a targeted-vs-random mask runner
4. a quality/drift readout attached to the intervention itself

This matters because the current repo can export activation tensors, but it cannot yet execute the exact intervention packet whose release we are reviewing.

So the white-box side is:

- `concept-ready`
- not `execution-ready`

## Review Axis 2: Gray-box Matched-Packet Surface

Current gray-box truth is strong at the split level:

- `PIA` runtime mainline is real
- defended pairs are real
- preview and mainline both run on real assets

But the current `I-C` packet needs something narrower:

- one matched member/control packet whose score-gap can be read against the same local samples used by the white-box intervention review

What is available now:

1. `runtime-preview-pia` proves real member/non-member batches can be loaded
2. `run-pia-runtime-mainline` emits split-level summary metrics

What is **not** yet frozen:

1. one explicit gray-box packet contract tied to the white-box canary pair
2. one repo-frozen per-sample `PIA` score-gap export on that packet
3. one targeted-vs-random comparison path on the same matched packet

This means the gray-box side is also:

- `surface-ready in principle`
- not `packet-ready in the exact I-C sense`

## Review Axis 3: GPU Value And Host Cost

If the packet were already executable, one bounded GPU rung could be justified.

But it is not honest to spend GPU merely to discover that:

1. the white-box intervention path was never wired
2. the gray-box packet-local score-gap was never frozen
3. the packet still needs CPU-side schema work before any result can be interpreted

Under the repo's current GPU policy, that is not a good release.

It would be:

- implementation-first GPU use
- not hypothesis-first GPU use

## Release Decision

The packet is **not** honest to release yet.

This is not because the hypothesis is dead.

It is because the repository still lacks the minimum executable surface for the exact packet it now claims to want.

So the correct reading is:

- `release blocked by missing execution surface`

More precisely, two blockers remain:

1. `white-box mask executor missing`
   - the repo has read-only export, not intervention execution
2. `gray-box matched-packet score export missing`
   - the repo has split-level summary and preview, not the exact packet-local bridge readout required by `I-C`

## What This Is Not

This is not:

- a falsification of `I-C`
- a demotion of the `I-C.1 / I-C.2 / I-C.3` packet design
- a claim that the next GPU question should move back to black-box, gray-box, or `DP-LoRA`

It is only a release-review verdict:

- `not yet honest to run`

## Verdict

- `cross_permission_bridge_packet_release_review_verdict = blocked`

More precise reading:

1. `I-C.4` is now satisfied:
   - the first hot-standby GPU candidate has been formally reviewed
2. current release decision:
   - `gpu_release = none`
   - `active_gpu_question = none`
   - `next_gpu_candidate = none`
3. reason:
   - missing white-box mask execution surface
   - missing gray-box matched-packet score-gap export surface

## Next Step

- `next_live_cpu_first_lane = I-C.5 minimum executable surface scaffolding for the white-gray bridge packet`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: light sync required
- `workspaces/white-box/plan.md`: update required
- `workspaces/implementation/challenger-queue.md`: light sync required
- `Leader/materials`: no sync needed; still below release-facing innovation wording
- `Platform/Runtime`: no direct handoff; the packet remains below executable surface
