# 2026-04-17 Cross-Permission Same-Packet And In-Model Intervention Contract

## Question

After `I-C.7` proved that the current white-box and gray-box CPU canaries are still two separate scaffolds, what exact contract should freeze:

1. same-packet identity,
2. packet size and alignment,
3. white-box intervention semantics,

so the next bridge task can be executed honestly without reopening theory drift or premature GPU release?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\plan.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-cross-permission-cpu-canary-interpretation.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\cross-permission-masked-packet-canary-20260417-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-packet-score-export-20260417-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\runs\pia-packet-score-export-20260417-r1\sample_scores.jsonl`
- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\gsa_observability.py`
- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\pia_adapter.py`
- `D:\Code\DiffAudit\Research\external\PIA\DDPM\CIFAR10_train_ratio0.5.npz`

## What The Current Scaffolds Actually Bind

### White-box side

Current white-box packet identity is:

- `split + sample_id`
- exactly one named member object
- exactly one named control object

Current canary pair:

- member:
  - `target-member/00-data_batch_1-00965.png`
- control:
  - `target-nonmember/00-data_batch_1-00467.png`

### Gray-box side

Current gray-box packet identity is:

- `mia_train_idxs / mia_eval_idxs` on the canonical `PIA` split
- a fixed index window
- current canary shape `4 member + 4 nonmember`

Current canary packet:

- members:
  - `10365 / 25853 / 41222 / 7409`
- nonmembers:
  - `6640 / 7423 / 45313 / 9057`

## New Binding Fact That Changes The Contract

The current white-box canary pair is not only size-misaligned with the gray-box packet.

It is also membership-misaligned under `PIA` semantics.

Using the CIFAR10 train-index implied by the white-box filenames:

- `target-member/00-data_batch_1-00965.png`
  - `canonical_index = 965`
- `target-nonmember/00-data_batch_1-00467.png`
  - `canonical_index = 467`

Checked against `PIA` split semantics:

- `965` is in `mia_train_idxs`
- `467` is also in `mia_train_idxs`
- `467` is **not** in `mia_eval_idxs`

So the current white-box “control/nonmember” object is not a valid gray-box nonmember under the canonical `PIA` split.

That means the next bridge task cannot inherit the current white-box pair unchanged and still call itself `same-packet`.

## Same-Packet Identity Contract

The first honest cross-permission packet is now frozen to the following identity rule.

### Canonical packet key

Use:

- `dataset = CIFAR10`
- `canonical_index`

Do not use as the primary key:

- raw `sample_id` alone
- directory name alone
- filename suffix alone

### Membership semantics

Membership must be defined by gray-box `PIA` split semantics:

- `member = mia_train_idxs`
- `nonmember = mia_eval_idxs`

The white-box directory label is now only an auxiliary locator.

It is not the authority for same-packet membership truth.

### Minimum binding fields

Each packet object must carry:

- `dataset`
- `canonical_index`
- `membership_semantics = pia_split_v1`
- `membership`
- `packet_position`
- `whitebox_split`
- `whitebox_sample_id`
- `graybox_split_index`

Recommended extra anti-drift field:

- `pixel_sha256`

### Alignment rule

The first honest bridge packet must contract to:

- `1 member + 1 nonmember`

Why:

1. the current white-box export surface is fundamentally a pair surface
2. `records_written = 2` on the current white-box scaffold
3. expanding gray-box downward to `1 + 1` is smaller and more honest than pretending the current white-box pair already matches a `4 + 4` packet

### Immediate consequence

The current white-box canary pair is not same-packet-ready.

Before any new bridge execution:

1. freeze one membership-consistent member object
2. freeze one membership-consistent nonmember object
3. bind both to `canonical_index`
4. assemble the gray-box packet from the same two canonical objects only

## In-Model Intervention Contract

The current offline tensor masking surface is now explicitly below the first bridge-execution contract.

The minimum honest in-model contract is:

- same `canonical packet`
- same `checkpoint`
- same `timestep`
- same `noise_seed`
- same `prediction_type`
- same `layer_selector = mid_block.attentions.0.to_v`
- same `mask_kind`
- same `channel_indices`
- same `alpha`

And the intervention must happen:

- inside the forward path
- before downstream model computation completes

It must not be satisfied by:

- exporting activations
- masking the saved tensor offline
- reading only layer-local pre/post deltas

### Required outputs

The minimum bridge-ready intervention packet must emit:

1. one `baseline` forward result
2. one `intervened` forward result
3. one white-box downstream readout
4. one gray-box packet-local readout on the same canonical packet

Minimum acceptable downstream reads:

- white-box:
  - one downstream activation or model-output drift reading
- gray-box:
  - `PIA member_control_score_gap` on the same `1 + 1` packet

## GPU Gate Decision

`I-C.8` does not restore GPU release.

GPU remains blocked until all of the following are true:

1. white-box and gray-box bind to the same `canonical_index` objects
2. packet size is explicitly frozen and identical on both sides
3. white-box intervention moves from offline masking to in-model execution
4. a CPU-first matched-packet run shows targeted white-box movement and gray-box weakening on the same packet, under the existing control and drift budget

Current honest posture therefore remains:

- `active_gpu_question = none`
- `next_gpu_candidate = none`

## Verdict

- `cross_permission_same_packet_intervention_contract_verdict = positive`

More precise reading:

1. `I-C.8` is now satisfied:
   - the repository has an explicit same-packet identity contract
   - the repository has an explicit in-model intervention contract
2. current execution truth becomes stricter, not looser:
   - the old white-box canary pair is no longer eligible to masquerade as the first matched bridge packet
3. this is still below bridge support:
   - no same-packet CPU co-movement has been executed yet
   - no GPU candidate is restored yet

## Next Step

- `next_live_cpu_first_lane = I-C.9 canonical-index bridge binding and membership-consistent pair freeze`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: light sync required
- `workspaces/white-box/plan.md`: update required
- `workspaces/implementation/challenger-queue.md`: optional light sync only after the first matched pair is frozen
- `Leader/materials`: no sync needed
- `Platform/Runtime`: no direct handoff; still below executed bridge truth
