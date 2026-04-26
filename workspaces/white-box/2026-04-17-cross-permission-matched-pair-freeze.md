# 2026-04-17 Cross-Permission Matched Pair Freeze

## Question

After `I-C.8` froze the same-packet identity contract, can the repository now bind one concrete `1 member + 1 nonmember` bridge pair across the white-box and gray-box scaffolds, using current admitted assets and current CPU-first export surfaces only?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-cross-permission-same-packet-intervention-contract.md`
- `D:\Code\DiffAudit\Research\external\PIA\DDPM\CIFAR10_train_ratio0.5.npz`
- `D:\Code\DiffAudit\Research\workspaces\white-box\assets\gsa-cifar10-1k-3shadow-epoch300-rerun1\datasets\target-member\`
- `D:\Code\DiffAudit\Research\workspaces\white-box\assets\gsa-cifar10-1k-3shadow-epoch300-rerun1\datasets\target-nonmember\`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\cross-permission-matched-pairfreeze-20260417-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-packet-score-export-matched-pairfreeze-20260417-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-packet-score-export-matched-pairfreeze-20260417-r1\sample_scores.jsonl`

## Selection Rule

The first matched bridge pair must satisfy all of:

1. member object exists in white-box `target-member`
2. nonmember object exists in white-box `target-nonmember`
3. member object maps to a `PIA` member index
4. nonmember object maps to a `PIA` nonmember index
5. pair shape remains `1 + 1`
6. no GPU execution is required

## Frozen Pair

The first matched pair is now frozen as:

- member object:
  - white-box sample:
    - `target-member/00-data_batch_1-00965.png`
  - `canonical_index = 965`
  - `PIA membership = member`
  - `PIA member_offset = 1238`
- nonmember object:
  - white-box sample:
    - `target-nonmember/00-data_batch_1-01278.png`
  - `canonical_index = 1278`
  - `PIA membership = nonmember`
  - `PIA nonmember_offset = 1803`

This freezes the first honest `same-packet` bridge pair under the `I-C.8` contract.

## CPU-First Artifact Check

### White-box side

Artifact:

- `workspaces/white-box/runs/cross-permission-matched-pairfreeze-20260417-r1/summary.json`

Observed result:

- `status = ready`
- member sample:
  - `target-member/00-data_batch_1-00965.png`
- control sample:
  - `target-nonmember/00-data_batch_1-01278.png`
- `selected_channel_abs_delta_pre = 0.047201`
- `selected_channel_abs_delta_post = 0.023601`
- `selected_delta_retention_ratio = 0.5`
- `off_mask_drift = 0.0`

Interpretation:

- current white-box scaffold can already emit the exact matched pair selected by `I-C.8`
- but it still remains an offline masking scaffold

### Gray-box side

Artifact:

- `workspaces/gray-box/runs/pia-packet-score-export-matched-pairfreeze-20260417-r1/summary.json`

Observed result:

- `status = ready`
- `packet_size = 1`
- `member_indices = [965]`
- `nonmember_indices = [1278]`
- `member_control_score_gap = -6.157752`

Interpretation:

- current gray-box scaffold can already emit packet-local `PIA` scores on the same `1 + 1` pair
- this is pair freeze only, not support

## What This Solves

This note solves:

1. the first concrete `same-packet` object freeze
2. the first exact gray-box offset binding for that pair
3. the first real CPU-first artifact pair on the same member/nonmember identity

## What This Does Not Solve

This note does not solve:

1. in-model intervention
2. white-gray co-movement support
3. targeted-vs-random control comparison on the matched pair
4. GPU re-release

The gray-box packet score gap is also currently negative on this pair.

That is not a support failure yet, because no matched-pair intervention comparison has been executed.

It only means:

- this pair is frozen as an executable object
- not that the hypothesis is supported

## Verdict

- `cross_permission_matched_pair_freeze_verdict = positive but bounded`

More precise reading:

1. `I-C.9` is now satisfied:
   - one membership-consistent `1 + 1` pair is frozen
   - both current scaffolds can already emit artifacts on that exact pair
2. the repository is now ready to move to intervention execution rather than more pair-selection discussion
3. GPU remains correctly blocked:
   - `active_gpu_question = none`
   - `next_gpu_candidate = none`

## Next Step

- `next_live_cpu_first_lane = I-C.10 implement the in-model white-box intervention surface and CPU matched-pair co-movement canary`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: light sync required
- `workspaces/white-box/plan.md`: update required
- `Leader/materials`: no sync needed
- `Platform/Runtime`: no direct handoff; still below executed intervention truth
