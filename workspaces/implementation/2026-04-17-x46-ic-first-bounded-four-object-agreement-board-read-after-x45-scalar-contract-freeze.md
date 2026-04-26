# 2026-04-17 X-46 I-C First Bounded Four-Object Agreement-Board Read After X-45 Scalar Contract Freeze

## Question

After `X-45` froze the white-box board-local scalar and the gray-box second pairboard score export is now ready, does the first honest `2 member + 2 nonmember` agreement board actually show enough directional agreement to count as a positive first read for the fresh `I-C` hypothesis?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\challenger-queue.md`
- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-17-x45-ic-whitebox-board-local-concentration-scalar-contract-freeze-after-x44-agreement-board-contract-review.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\cross-permission-board-local-scalar-probe-20260417-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-packet-score-export-matched-pairfreeze-20260417-r1\sample_scores.jsonl`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-packet-score-export-secondary-pairboard-20260417-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-packet-score-export-secondary-pairboard-20260417-r1\sample_scores.jsonl`

## Frozen Board Read

The first bounded agreement board is now:

- member `965`
- nonmember `1278`
- member `8`
- nonmember `23`

White-box scalar:

- `selected_channel_abs_profile_mean`

Gray-box scalar:

- raw per-object `PIA` `score` from `sample_scores.jsonl`

Higher-is-stronger reading is kept unchanged on both sides:

- larger white-box scalar = stronger localized magnitude on the frozen selected channels
- larger gray-box score = more positive `PIA` score on the fixed one-object packet export

## Observed Board

### White-box descending order

- `965`
- `1278`
- `8`
- `23`

### Gray-box descending order

- `1278`
- `8`
- `965`
- `23`

### Per-object values

- `965`:
  - white-box `0.730977`
  - gray-box `-18.348583`
- `1278`:
  - white-box `0.710589`
  - gray-box `-12.190831`
- `8`:
  - white-box `0.709006`
  - gray-box `-12.244166`
- `23`:
  - white-box `0.704266`
  - gray-box `-38.920979`

### Aggregate alignment

- white-box member mean:
  - `0.719991`
- white-box nonmember mean:
  - `0.707427`
- gray-box member mean:
  - `-15.296374`
- gray-box nonmember mean:
  - `-25.555905`
- rank correlation on the four-object board:
  - `Spearman = 0.4`

## Verdict Read

What is encouraging:

1. both surfaces still put members above nonmembers on average;
2. both surfaces agree on the weakest object:
   - `23`
3. the board does not collapse into random noise.

What is not good enough:

1. the top object is not shared:
   - white-box top = `965`
   - gray-box top = `1278`
2. the top-three object ordering is materially different;
3. the first bounded board therefore does not deliver the clean same-object broad-order agreement that `X-41` was trying to get before any stronger support language.

## Result

The first bounded four-object agreement-board read is real, but it is not clean enough to count as positive agreement-first support.

The fresh `I-C` line should therefore not keep the main slot by default on this packet.

## Selection

- `selected_next_live_lane = X-47 non-graybox next-lane reselection after X-46 first bounded agreement-board read`

## Verdict

- `x46_ic_first_bounded_four_object_agreement_board_read_after_x45_scalar_contract_freeze = negative but useful`

More precise reading:

1. the fresh `I-C` hypothesis did receive a first honest bounded board read;
2. that read contains partial same-direction residue, but not a clean enough object-order agreement to count as support;
3. the current `I-C` line should now yield the next main-slot decision back to non-graybox reselection unless a stronger new bounded hypothesis appears.

## Frozen Posture

- `active_gpu_question = none`
- `next_gpu_candidate = none`
- `completed_task = X-46 I-C first bounded four-object agreement-board read after X-45 scalar contract freeze`
- `next_live_cpu_first_lane = X-47 non-graybox next-lane reselection after X-46 first bounded agreement-board read`
- `carry_forward_cpu_sidecar = I-A higher-layer boundary maintenance`

## Canonical Evidence Anchor

- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-17-x46-ic-first-bounded-four-object-agreement-board-read-after-x45-scalar-contract-freeze.md`

## Handoff Decision

- `D:\Code\DiffAudit\Research\ROADMAP.md`: update required
- `D:\Code\DiffAudit\Research\workspaces\implementation\challenger-queue.md`: update required
- `D:\Code\DiffAudit\Research\docs\comprehensive-progress.md`: note-level sync recommended
- `D:\Code\DiffAudit\Research\docs\mainline-narrative.md`: note-level sync recommended
- `D:\Code\DiffAudit\ROADMAP.md`: update required
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this changes live lane selection and the fresh `I-C` branch reading, but it still does not change admitted metrics, exported consumer schema, or runtime requirements.
