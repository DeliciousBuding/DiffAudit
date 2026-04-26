# 2026-04-17 Cross-Permission Executable Surface Scaffolding

## Question

After the `I-C.4` release review closed as `blocked`, what is the minimum executable surface scaffolding the repo must add before one bounded white-gray bridge packet could be judged honestly?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-cross-permission-bridge-packet-release-review.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-cross-permission-support-contract.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-cross-permission-mask-selection.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-finding-nemo-bounded-local-intervention-proposal.md`
- `D:\Code\DiffAudit\Research\workspaces\gray-box\2026-04-16-cdi-protocol-asset-contract.md`
- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\gsa_observability.py`
- `D:\Code\DiffAudit\Research\src\diffaudit\attacks\pia_adapter.py`
- `D:\Code\DiffAudit\Research\src\diffaudit\cli.py`

## Core Principle

The missing pieces are not another theory note.

They are two concrete execution surfaces:

1. a white-box local mask executor
2. a gray-box matched-packet score exporter

Without both, any future GPU run would still be implementation work disguised as a hypothesis test.

## Scaffold A: White-box Masked Packet Executor

### Purpose

Execute one targeted or control mask on the already-frozen white-box canary pair and emit both pre-mask and post-mask local readings.

### Minimum command shape

The first executable scaffold should be a new command family, conceptually:

- `export-gsa-observability-masked-packet`

### Minimum inputs

1. admitted `checkpoint_root`
2. fixed `sample_id`
3. fixed `control_sample_id`
4. fixed `layer_selector`
5. fixed `timestep`
6. mask spec:
   - `mask_name`
   - `channel_indices`
   - `alpha`

### Minimum outputs

1. `summary.json`
2. `records.jsonl`
3. `pre_mask_tensor`
4. `post_mask_tensor`
5. derived packet fields:
   - `selected_channel_abs_delta`
   - `selected_delta_retention_ratio`
   - `off_mask_drift`

### Boundary

This scaffold should stay:

- CPU-first for the first rung
- selector-local
- timestep-local
- channel-local

It must not silently expand into:

- neuron discovery
- multi-layer intervention
- benchmark release

## Scaffold B: Gray-box Matched-Packet Score Exporter

### Purpose

Emit one machine-readable `PIA` packet-local score bundle for a fixed matched member/control packet, so the white-box intervention can be compared against a real gray-box readout rather than only split-level aggregates.

### Minimum command shape

The first executable scaffold should be a new command family, conceptually:

- `export-pia-packet-scores`

### Minimum inputs

1. current canonical `PIA` config
2. canonical checkpoint root
3. canonical member split root
4. one fixed packet size
5. explicit packet membership:
   - member indices
   - non-member indices
6. optional defense toggle

### Minimum outputs

1. `summary.json`
2. `sample_scores.jsonl`
3. packet-level fields:
   - `member_score_mean`
   - `control_score_mean`
   - `member_control_score_gap`
4. reproducibility fields:
   - `member_indices`
   - `nonmember_indices`
   - `attack_num`
   - `interval`
   - `batch_size`

### Boundary

The first exporter does not need to solve:

- full split-level benchmark refresh
- adaptive attacker retraining
- new defense family evaluation

It only needs to make the `I-C` packet readable on the gray-box side.

## Smallest Honest Build Order

The minimum honest order is now:

1. implement white-box masked packet executor on CPU
2. implement gray-box matched-packet score exporter on CPU
3. verify both emit stable machine-readable artifacts
4. only then reconsider one bounded GPU bridge packet

If step 1 or step 2 fails, the packet stays below release.

## Why This Is Enough

This scaffolding is sufficient because it directly addresses both blockers from `I-C.4`:

1. it turns white-box intervention from proposal into executable packet
2. it turns gray-box bridge reading from summary-level story into packet-level artifact

Anything larger is optional later work, not minimum unblock work.

## Verdict

- `cross_permission_executable_surface_scaffolding_verdict = positive but bounded`

More precise reading:

1. `I-C.5` is now satisfied:
   - the minimum unblock surface has been frozen
2. this is still pre-implementation:
   - no new code yet
   - no CPU canary yet
   - no GPU release yet
3. the next honest task is now implementation-scaffold oriented, not theory-oriented

## Next Step

- `next_live_cpu_first_lane = I-C.6 implement the minimum CPU-first scaffold for the white-gray bridge packet`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: light sync required
- `workspaces/white-box/plan.md`: update required
- `workspaces/implementation/challenger-queue.md`: light sync required
- `Leader/materials`: no sync needed
- `Platform/Runtime`: no direct handoff; still pre-implementation
