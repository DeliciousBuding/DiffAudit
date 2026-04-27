# 2026-04-17 Cross-Permission Translated Falsifier Review

## Question

After `I-C.13` froze the bridge promotion boundary at `translated-contract canary only`, does the first translated-contract targeted-vs-random falsifier on the frozen pair `965 / 1278` now show that the targeted mask beats same-budget controls honestly enough to reopen support discussion?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-cross-permission-bridge-verdict-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-translated-alias-probe-20260417-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-translated-alias-probe-random-20260417-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-translated-alias-probe-bottom-20260417-r1/summary.json`

## Executed Comparison

Common fixed contract:

- alias selector:
  - `middleblocks.0.attn.proj_v`
- translated from:
  - `mid_block.attentions.0.to_v`
- translation kind:
  - `translated-contract`
- same-spec reuse:
  - `false`
- tensor layout:
  - `BCHW`
- channel axis:
  - `channel_dim = 1`
- packet:
  - member `965`
  - nonmember `1278`
- locality budget:
  - `k = 8`
  - `alpha = 0.5`

Compared masks:

1. targeted:
   - `top_abs_delta_k`
2. control A:
   - `random_k_seeded`
3. control B:
   - `bottom_abs_delta_k`

## Observed Results

### Targeted mask

- `selected_channel_abs_delta_pre = 1.025205`
- `selected_channel_abs_delta_post = 0.512603`
- `selected_delta_retention_ratio = 0.5`
- `member_control_score_gap_delta = -0.033422`

### Random control

- `selected_channel_abs_delta_pre = 0.273676`
- `selected_channel_abs_delta_post = 0.136838`
- `selected_delta_retention_ratio = 0.5`
- `member_control_score_gap_delta = 0.031760`

### Bottom control

- `selected_channel_abs_delta_pre = 0.003453`
- `selected_channel_abs_delta_post = 0.001726`
- `selected_delta_retention_ratio = 0.5`
- `member_control_score_gap_delta = -0.003209`

## Reading

### 1. Targeted mask is stronger locally than both controls

On the alias-local contrast itself, targeted is clearly not random noise:

- targeted pre-delta is much larger than random
- targeted pre-delta is vastly larger than bottom

So the translated contract is not collapsing into a trivial mask-selection failure.

### 2. But targeted still does not beat random in support direction

The key bridge question is not:

- does anything move?

It is:

- does the targeted mask produce the more support-eligible gray-box movement?

Here the answer is no.

Observed packet-local gray-box direction:

- targeted drives the already-negative gap further negative
- random moves the gap in the opposite direction and with similar magnitude
- bottom stays near-flat

So the targeted mask does not beat the matched random control on the gray-box support-facing readout.

### 3. Therefore the first translated-contract falsifier fails on this surface

This does not mean:

- the whole `I-C` track is impossible

But it does mean:

- this first translated-contract bridge packet does not earn support
- the current frozen pair and current alias contract do not justify promotion

## Verdict

- `cross_permission_translated_falsifier_review_verdict = negative but useful`

More precise reading:

1. the first translated-contract falsifier fails on the current frozen pair
2. targeted alias-local movement is real
3. but the targeted gray-box packet effect does not beat matched random in the support-facing direction
4. `I-C` therefore remains below support and below GPU release

## Consequence For The Track

Current honest reading for `I-C` is now:

- same-packet identity exists
- white-box in-model execution exists
- gray-box translated-contract execution exists
- first translated-contract falsifier is negative on the current frozen pair

This is enough to justify:

- freezing the current `I-C` packet below support

This is not enough to justify:

- `support`
- GPU release
- stronger cross-box headline wording

## Next Step

- `next_live_cpu_first_lane = X-12 non-graybox next-lane reselection after translated I-C falsifier`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

Why the lane should yield now:

1. the current translated `I-C` packet has produced both its first positive executability result and its first negative falsifier result
2. continuing to iterate on the same frozen pair without a new hypothesis would now risk low-value same-family churn
3. the next highest-value action is to reseat the main lane rather than keep forcing this packet family

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: light sync required
- `workspaces/white-box/plan.md`: update required
- `Leader/materials`: only if higher-layer wording tries to write `I-C` stronger than `translated-contract-only + negative falsifier`
- `Platform/Runtime`: no direct handoff
