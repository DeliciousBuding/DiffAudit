# 2026-04-17 X-45 I-C White-Box Board-Local Concentration Scalar Contract Freeze After X-44 Agreement-Board Contract Review

## Question

After `X-44` narrowed the remaining blocker to white-box contract definition, can the repository now freeze one honest board-local concentration scalar and selector policy for the fresh `I-C` `2 member + 2 nonmember` board?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-x44-ic-bounded-multipair-agreement-board-contract-freeze-after-x43-pairboard-identity-freeze.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-finding-nemo-bounded-localization-observable-selection.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-cross-permission-mask-selection.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-17-cross-permission-matched-pair-freeze.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/runs/cross-permission-matched-pairfreeze-20260417-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/runs/cross-permission-board-local-scalar-probe-20260417-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/runs/cross-permission-board-local-scalar-probe-20260417-r1/rows.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/pia-packet-score-export-secondary-pairboard-20260417-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/gsa_observability.py`

## Candidate Review

### 1. Reuse `summary_stat` as the white-box board scalar

Not selected.

Reason:

1. the already-admitted localization boundary explicitly says `summary_stat` is sanity metadata rather than a localization score;
2. its values are too compressed to carry the board-local ranking burden honestly;
3. promoting it now would directly violate the existing white-box boundary contract.

### 2. Use one full-channel global magnitude scalar

Not selected.

Candidate shape:

- mean absolute value over all channels in `_channelwise_profile(...)`

Reason:

1. this is object-level, but it is not board-local;
2. it ignores the already-frozen selected-channel locality object from the current `I-C` selector family;
3. on the bounded four-object probe it stays nearly flat:
   - `0.488650 / 0.489374 / 0.489370 / 0.488756`
4. that makes it a weak global activity proxy rather than the intended localized concentration read.

### 3. Use selected-channel absolute profile mean on one frozen board-wide channel set

Selected.

Definition:

1. capture the raw activation tensor on the frozen surface:
   - selector `mid_block.attentions.0.to_v`
   - timestep `999`
   - noise seed `7`
   - admitted `GSA epoch300 rerun1` target checkpoint root
2. reduce each object to one per-channel profile via:
   - `_channelwise_profile(activation)`
3. freeze one board-wide channel set by reusing the already-frozen pair-A selected channels from:
   - `cross-permission-matched-pairfreeze-20260417-r1`
4. define the scalar as:
   - `selected_channel_abs_profile_mean = mean(abs(profile[selected_channels]))`

## Frozen Selector Policy

The first honest board-wide selector policy is now frozen as:

- selector:
  - `mid_block.attentions.0.to_v`
- timestep:
  - `999`
- noise seed:
  - `7`
- checkpoint family:
  - admitted `gsa-cifar10-1k-3shadow-epoch300-rerun1` target root
- board-wide selected channels:
  - `[5, 471, 1, 135, 360, 215, 394, 425]`
- channel source:
  - the already-frozen pair-A matched-pair selector from `965 / 1278`

Why this remains honest:

1. the selected channels are frozen before board interpretation, not re-picked per object;
2. the board uses one fixed localized surface rather than a new search over selectors or timesteps;
3. the scalar remains derived from raw tensor artifacts, not from pair-local deltas.

## Frozen White-Box Board Scalar

The first honest white-box board-local concentration scalar is now frozen as:

- `selected_channel_abs_profile_mean`

Interpretation boundary:

1. it is an object-level localized magnitude read on one already-frozen channel set;
2. it is not a neuron claim;
3. it is not a causal score;
4. it is not by itself a support verdict;
5. it is one board-local white-box scalar suitable for the next agreement-board read.

## Bounded Verification

On the bounded four-object board:

- pair A member `965`:
  - `0.730977`
- pair A nonmember `1278`:
  - `0.710589`
- pair B member `8`:
  - `0.709006`
- pair B nonmember `23`:
  - `0.704266`

Aggregate read:

- member mean:
  - `0.719991`
- nonmember mean:
  - `0.707427`
- mean gap:
  - `0.012564`

This is enough to show:

1. the scalar is not degenerate on the current board;
2. the scalar preserves object-level variation on the frozen localized surface;
3. but this remains a contract freeze, not agreement support.

## Supplementary Preparation Cleared

The next board read is now operationally ready on the gray-box side as well:

- secondary pairboard score export:
  - member `8` score `-12.244166`
  - nonmember `23` score `-38.920979`

This does not change the `X-45` verdict.

It only means the next board read no longer needs another gray-box prep step first.

## Selection

- `selected_next_live_lane = X-46 I-C first bounded four-object agreement-board read after X-45 scalar contract freeze`

## Verdict

- `x45_ic_whitebox_board_local_concentration_scalar_contract_freeze_after_x44_agreement_board_contract_review = positive but bounded`

More precise reading:

1. the white-box blocker identified by `X-44` is now resolved;
2. the fresh `I-C` line now has one honest object-level localized white-box scalar;
3. the next step can return to board reading rather than scalar-definition churn;
4. GPU still remains correctly idle.

## Frozen Posture

- `active_gpu_question = none`
- `next_gpu_candidate = none`
- `completed_task = X-45 I-C white-box board-local concentration scalar contract freeze after X-44 agreement-board contract review`
- `next_live_cpu_first_lane = X-46 I-C first bounded four-object agreement-board read after X-45 scalar contract freeze`
- `carry_forward_cpu_sidecar = I-A higher-layer boundary maintenance`

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-17-x45-ic-whitebox-board-local-concentration-scalar-contract-freeze-after-x44-agreement-board-contract-review.md`

## Handoff Decision

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`: update required
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/challenger-queue.md`: update required
- `<DIFFAUDIT_ROOT>/Research/docs/comprehensive-progress.md`: update required
- `<DIFFAUDIT_ROOT>/Research/docs/mainline-narrative.md`: update required
- `<DIFFAUDIT_ROOT>/ROADMAP.md`: update required
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this resolves a real blocker for the fresh `I-C` line, but it still does not alter admitted metrics, runtime consumer fields, or external competition wording.
