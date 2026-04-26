# 2026-04-17 Finding NeMo First Truly Bounded Admitted Packet Launch Review

## Question

After `I-B.12` cleared extraction-side boundedness, is one concrete admitted `GSA` target-anchored fixed-mask intervention-on/off packet now honest and host-fit enough to occupy the next single GPU slot?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\ROADMAP.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-finding-nemo-first-intervention-on-off-bounded-review-contract-selection.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-finding-nemo-first-admitted-fixed-mask-packet-execution-budget-review.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-finding-nemo-extraction-side-bounded-cap-verdict.md`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\finding-nemo-first-bounded-localization-packet-20260417-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-runtime-mainline-20260409-cifar10-1k-3shadow-epoch300-rerun1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\white-box\assets\gsa-cifar10-1k-3shadow-epoch300-rerun1\manifests\cifar10-ddpm-1k-3shadow-epoch300-rerun1.json`
- live host check via `nvidia-smi` on `2026-04-17 06:48`

## Frozen Launch Config

The first honest admitted packet is released with no contract drift:

- command surface:
  - `run-gsa-runtime-intervention-review`
- source attack family:
  - admitted `GSA epoch300 rerun1`
- frozen mask summary:
  - `D:\Code\DiffAudit\Research\workspaces\white-box\runs\finding-nemo-first-bounded-localization-packet-20260417-r1\summary.json`
- workspace:
  - `D:\Code\DiffAudit\Research\workspaces\white-box\runs\finding-nemo-first-truly-bounded-admitted-intervention-review-20260417-r1`
- device:
  - `cuda`
- bounded packet size:
  - `max_samples = 64`
  - `extraction_max_samples = 64`
- paper-aligned runtime fields:
  - `resolution = 32`
  - `ddpm_num_steps = 1000`
  - `sampling_frequency = 10`
  - `attack_method = 1`
  - `prediction_type = epsilon`

## Honesty Review

### 1. The packet stays on the already frozen `I-B.9` contract

Nothing is reselected here:

- same `max_samples = 64` bounded board
- same selector `mid_block.attentions.0.to_v`
- same timestep `999`
- same `top_abs_delta_k` mask
- same `k = 8`
- same `alpha = 0.5`
- same frozen channels:
  - `[374, 471, 269, 1, 62, 360, 187, 394]`

So the launch does not quietly widen into:

- full-board execution
- per-model mask reselection
- new checkpoint-side identity
- new evaluation surface

### 2. The packet is now truly bounded at both layers

With the admitted manifest, one full board is:

- `1000 + 1000 + 3 * (1000 + 1000) = 8000` images

The first dual-run admitted packet is now released as:

- `8` splits
- `64` extracted images per split
- `2` branches (`baseline + intervened`)
- total extraction budget:
  - `8 * 64 * 2 = 1024` images

This is the key honesty change relative to `I-B.11`:

- no more hidden `16000`-image extraction budget
- the first real packet is bounded in both extraction and evaluation

### 3. The host-fit check is positive with a cost warning, not a no-go

Observed host state before launch:

- GPU:
  - `RTX 4070 8GB`
- live compute jobs:
  - none
- visible occupied memory:
  - about `2398 MiB / 8188 MiB`

This does **not** prove the run is cheap.

But it does support the more honest reading:

1. the only active GPU slot is currently free;
2. the bounded packet is small enough to justify occupying it;
3. the packet still deserves a cost warning because it replays the full paper-aligned timestep schedule twice.

## Verdict

- `finding_nemo_first_truly_bounded_admitted_packet_launch_review_verdict = positive`

More precise reading:

1. `I-B.13` is satisfied:
   - one concrete admitted launch config is now frozen as honest and host-fit
2. this is still not a defense verdict:
   - it is a launch-release verdict
3. the next honest action is no longer another CPU review:
   - it is the actual bounded packet execution on the single active GPU slot

## Canonical Evidence Anchor

- `D:\Code\DiffAudit\Research\workspaces\white-box\2026-04-17-finding-nemo-first-truly-bounded-admitted-packet-launch-review.md`

## Next Step

- `active_gpu_question = I-B.13 actual target-anchored fixed-mask intervention-on/off bounded packet on admitted GSA assets`
- `next_gpu_candidate = none while the packet is active`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: update required
- `workspaces/implementation/challenger-queue.md`: update required
- `workspaces/white-box/plan.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
