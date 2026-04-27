# 2026-04-17 Cross-Permission Selector-Alias Compatibility Review

## Question

After `I-C.10` landed a real white-box in-model intervention surface, is there now one honest gray-box selector alias compatible enough with the current white-box contract to justify the next bridge task, or does the bridge still block on architecture mismatch?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-cross-permission-inmodel-intervention-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-cross-permission-matched-pair-freeze.md`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/gsa_observability.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/pia_adapter.py`
- `<DIFFAUDIT_ROOT>/Research/external/PIA/DDPM/model.py`
- runtime selector probe on the frozen matched pair `965 / 1278`

## White-Box Selector Truth

Current frozen white-box selector:

- `mid_block.attentions.0.to_v`

Current frozen white-box parameter shape:

- `mid_block.attentions.0.to_v.weight`
  - `(512, 512)`

Current white-box channel contract:

- profile uses the last tensor axis as `channel`
- mask application also assumes the selected indices live on the last tensor axis

That contract is still honest for the current `GSA` surface.

## Gray-Box Alias Candidates

### Primary candidate

- `middleblocks.0.attn.proj_v`

Why this is the first honest alias:

1. it is the only `PIA` module that is both:
   - in the middle block, and
   - the attention value projection
2. it preserves the closest structural meaning to the white-box selector

Observed runtime hook shape on the frozen matched pair:

- `965`:
  - alias tensor shape `= (1, 256, 4, 4)`
- `1278`:
  - alias tensor shape `= (1, 256, 4, 4)`

### Secondary candidate

- `middleblocks.0.attn`

This is a weaker but still honest fallback only if the project explicitly widens from:

- value projection alias

to:

- block-level attention alias

It is not same-selector reuse.

### Rejected as first alias

- `downblocks.3.attn.proj_v`
- `downblocks.4.attn.proj_v`
- `upblocks.8.attn.proj_v`
- `upblocks.9.attn.proj_v`
- `upblocks.10.attn.proj_v`

These are family-level similar modules, but they are not the closest honest counterpart to the white-box `mid_block` selector.

## Strong Blockers

### 1. Axis semantics mismatch

White-box current contract:

- `channel_dim = last axis`

Primary gray-box alias runtime truth:

- tensor layout is `B, C, H, W`
- `channel_dim = 1`

So the current white-box `channel_indices` contract cannot be directly reused on the gray-box alias.

### 2. Width and parameter-shape mismatch

White-box selector:

- `(512, 512)`

Primary gray-box alias:

- `(256, 256, 1, 1)`

This is not a naming-only mismatch.

It means:

1. different width
2. different operator family
3. different channel namespace

So direct same-index reuse would be fake same-spec behavior.

### 3. Current support contract is still below alias translation

The first `I-C` packet currently freezes:

- one selector
- one channel mask family
- one locality budget

But it does **not** yet freeze:

- an admitted translated alias rule for another architecture

That translation rule has to be written before a gray-box intervention can be called bridge-compatible.

## What Is Not A Hard Blocker

The following are engineering tasks, not theory no-go:

1. adding a gray-box hook at `middleblocks.0.attn.proj_v`
2. exporting alias activations on the matched pair
3. making profile/mask logic accept an explicit `channel_dim`
4. rerunning `PIA` packet scores under a gray-box alias-scoped intervention

Those tasks are real work, but they are not themselves evidence that the bridge is conceptually invalid.

## Verdict

- `cross_permission_selector_alias_compatibility_review_verdict = blocked but useful`

More precise reading:

1. the bridge is not `no-go`
   - there is one honest primary alias candidate
2. the bridge is not yet `positive`
   - current white-box selector and channel contract still cannot be reused directly on gray-box
3. the blocker is now explicit:
   - alias translation rule
   - channel-axis translation
   - architecture compatibility

## Next Step

- `next_live_cpu_first_lane = I-C.12 gray-box translated-contract alias probe on middleblocks.0.attn.proj_v`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: light sync required
- `workspaces/white-box/plan.md`: update required
- `Leader/materials`: no sync needed
- `Platform/Runtime`: no direct handoff; still below stable export-contract change
