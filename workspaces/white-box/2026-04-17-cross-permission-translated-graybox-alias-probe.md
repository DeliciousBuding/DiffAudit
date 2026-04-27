# 2026-04-17 Cross-Permission Translated Gray-Box Alias Probe

## Question

After `I-C.11` froze `middleblocks.0.attn.proj_v` as the first honest gray-box alias, can the repository now execute one CPU-only translated-contract canary on the frozen matched pair `965 / 1278`, without pretending same-spec reuse?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-cross-permission-selector-alias-compatibility-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-cross-permission-inmodel-intervention-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-cross-permission-matched-pair-freeze.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-packet-score-export-matched-pairfreeze-20260417-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/pia_adapter.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/cli.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_pia_adapter.py`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-translated-alias-probe-20260417-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-translated-alias-probe-20260417-r1/sample_scores.jsonl`

## Executed Contract

The executed gray-box canary is:

- alias selector:
  - `middleblocks.0.attn.proj_v`
- translated from:
  - `mid_block.attentions.0.to_v`
- translation kind:
  - `translated-contract`
- same-spec reuse:
  - `false`
- tensor layout:
  - `B, C, H, W`
- channel axis:
  - `channel_dim = 1`
- frozen packet:
  - member `canonical_index = 965`
  - nonmember `canonical_index = 1278`
- mask:
  - `top_abs_delta_k`
  - `k = 8`
  - `alpha = 0.5`
- device:
  - `cpu`

Command surface:

- `export-pia-translated-alias-probe`

Canonical run artifact:

- `workspaces/gray-box/runs/pia-translated-alias-probe-20260417-r1/summary.json`

## Observed Runtime Facts

From the executed canary:

- `status = ready`
- `selector_resolved = true`
- `alias_weight_shape = (256, 256, 1, 1)`
- `alias_activation_shape = (1, 256, 4, 4)`
- score-path hook hits:
  - baseline `= 62`
  - intervened `= 62`
- forward-only hook hits:
  - baseline `= 2`
  - intervened `= 2`

Packet score movement:

- baseline member score:
  - `-18.348583`
- baseline nonmember score:
  - `-12.190831`
- baseline member-control gap:
  - `-6.157752`
- intervened member score:
  - `-18.382414`
- intervened nonmember score:
  - `-12.191239`
- intervened member-control gap:
  - `-6.191175`
- gap delta:
  - `-0.033422`

Alias-local mask movement:

- `selected_channel_abs_delta_pre = 1.025205`
- `selected_channel_abs_delta_post = 0.512603`
- `selected_delta_retention_ratio = 0.5`
- `off_mask_drift = 0.0`

Sample-level score deltas:

- member `965`:
  - `score_delta = -0.033831`
- nonmember `1278`:
  - `score_delta = -0.000408`

## Reading

This closes one real gap:

1. gray-box translated alias execution is now real rather than hypothetical
2. the repository now has one honest `BCHW + channel_dim=1` contract instead of silently reusing the white-box last-axis contract
3. the same frozen `965 / 1278` pair now produces a nonzero gray-box packet delta under alias-scoped intervention

This does **not** close the larger bridge question:

1. it is still not same-spec reuse
2. it is still not targeted-vs-random bridge evidence
3. it is still not enough to promote `I-C` to support wording
4. it still does not justify GPU release

## Verdict

- `cross_permission_translated_graybox_alias_probe_verdict = positive but bounded`

More precise reading:

1. `I-C.12` is satisfied:
   - the translated gray-box alias probe exists
   - it runs on the frozen matched pair
   - it records explicit translation boundaries instead of hiding them
2. the effect is real but small:
   - packet-local `PIA` scores move
   - the member moves more than the nonmember on this pair
3. the result stays below support:
   - this is a translated-contract canary only
   - it is not a same-spec white-gray bridge confirmation

## Next Step

- `next_live_cpu_first_lane = I-C.13 bridge verdict review after translated-contract canary`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

The next review question should be:

- does the executed `I-C.10 + I-C.12` packet set justify any stronger bridge verdict than `hypothesis survives as translated-contract only`, or should `I-C` remain below support with `gpu_release = none`?

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: light sync required
- `workspaces/white-box/plan.md`: update required
- `Leader/materials`: no sync needed
- `Platform/Runtime`: no direct handoff; current result is still below stable consumer-contract change
