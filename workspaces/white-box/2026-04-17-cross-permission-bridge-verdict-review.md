# 2026-04-17 Cross-Permission Bridge Verdict Review

## Question

After `I-C.10` landed a real white-box in-model intervention surface and `I-C.12` landed a real gray-box translated-contract alias probe on the same frozen pair `965 / 1278`, does the executed packet set now deserve any stronger bridge verdict than `translated-contract canary only`?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-cross-permission-inmodel-intervention-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-cross-permission-selector-alias-compatibility-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-cross-permission-translated-graybox-alias-probe.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/runs/cross-permission-inmodel-packet-canary-20260417-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-translated-alias-probe-20260417-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-translated-alias-probe-20260417-r1/sample_scores.jsonl`

## Support Contract Reminder

The current `I-C` support threshold is still stricter than:

1. white-box local movement exists
2. gray-box translated alias execution exists

The first honest support tier still requires all of:

1. one matched packet identity
2. one executed white-box intervention
3. one gray-box packet-local directional movement that can honestly count as weakening the member signal on that same packet
4. a future targeted-vs-random control comparison under the same locality budget

The current packet set only satisfies the first two items fully.

## What The Executed Packet Set Now Proves

### White-box side

`I-C.10` now proves:

- the matched pair can be rerun with the intervention applied inside the forward path
- local selected-channel contrast shrinks as intended
- one downstream `epsilon`-prediction drift readout exists

Observed white-box downstream reading:

- `selected_delta_retention_ratio = 0.5`
- `off_mask_drift = 0.0`
- `epsilon_prediction_rms_drift_mean = 2.78113e-07`
- `prediction_drift_gap = -8.38e-10`

### Gray-box side

`I-C.12` now proves:

- the first honest gray-box alias can be executed as a translated contract
- the translation is explicit:
  - `alias_selector = middleblocks.0.attn.proj_v`
  - `translated_from = mid_block.attentions.0.to_v`
  - `same_spec_reuse = false`
  - `tensor_layout = BCHW`
  - `channel_dim = 1`
- packet-local `PIA` scores move non-trivially on the frozen pair

Observed gray-box packet reading:

- baseline member-control gap:
  - `-6.157752`
- intervened member-control gap:
  - `-6.191175`
- gap delta:
  - `-0.033422`
- sample-level deltas:
  - member `965`:
    - `-0.033831`
  - nonmember `1278`:
    - `-0.000408`

## Why This Still Does Not Earn Support

### 1. The gray-box packet still is not support-eligible by direction

The frozen pair does not start from a positive gray-box member advantage.

Instead:

- baseline member-control gap is already negative

So the current gray-box packet does not demonstrate:

- a member advantage that weakens toward zero under the shared packet intervention

It only demonstrates:

- that translated-contract execution changes the packet scores

That is useful execution truth, but not support truth.

### 2. The bridge still lacks targeted-vs-random falsifier execution

`I-C.1` did not ask merely for one targeted run.

It asked for:

- targeted mask movement that beats a matched random control

No such targeted-vs-random translated bridge result exists yet on the executed packet set.

So the support contract is still incomplete even if the current directional reading had looked cleaner.

### 3. Same-spec reuse is still false

The executed gray-box result is an admitted translated contract.

It is explicitly not:

- same selector reuse
- same axis semantics reuse
- same operator-family reuse

That means the current packet can raise confidence in executability, but not collapse the architecture boundary.

### 4. White-box downstream drift is still too weak to carry the bridge alone

The white-box in-model packet is useful because it proves the intervention really runs inside the forward path.

But its downstream drift remains extremely small.

So the white-box side alone still cannot carry a stronger cross-box reading when the gray-box side is only translated-contract and not direction-clean.

## Verdict

- `cross_permission_bridge_verdict_review_verdict = blocked`

More precise reading:

1. no stronger bridge verdict than `translated-contract canary only` is currently justified
2. `I-C` should remain hypothesis-only and below support wording
3. the block is promotion-facing, not execution-facing:
   - executability improved
   - but the first honest support tier is still unclosed
4. this is not a full-track `no-go`
   - the architecture boundary is now sharper
   - the track may still reopen under a tighter falsifier
5. this is not a GPU-release event
   - `gpu_release = none`
   - `next_gpu_candidate = none`

## Consequence For The Track

Current honest status for `I-C` is now:

- one real white-box in-model packet
- one real gray-box translated-contract packet
- no first support-tier bridge verdict yet

Current hold rule:

- do not reopen GPU for `I-C` unless a genuinely new bounded hypothesis appears

Examples of acceptable future re-open reasons:

1. one translated-contract targeted-vs-random packet on the same identity contract
2. one bridge-ready packet where the gray-box directional metric is support-eligible at baseline
3. one stronger same-surface contract that materially reduces translation ambiguity

## Next Step

- `next_live_cpu_first_lane = I-C.14 translated-contract targeted-vs-random falsifier on the frozen pair`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

The next `I-C` step should stay narrow:

1. keep the same frozen `965 / 1278` identity contract
2. keep the same translated alias semantics
3. add the missing same-budget controls:
   - `random_k_seeded`
   - `bottom_abs_delta_k`
4. only then ask whether the translated packet is still directionally better than control

What should not happen next:

1. do not promote this result into system-level support wording yet
2. do not reopen GPU
3. do not fallback into vague maintenance-only work while the sharper falsifier is still visible

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: light sync required
- `workspaces/white-box/plan.md`: update required
- `Leader/materials`: no sync needed
- `Platform/Runtime`: no direct handoff; current result sharpens a boundary but does not yet change a stable consumer contract
