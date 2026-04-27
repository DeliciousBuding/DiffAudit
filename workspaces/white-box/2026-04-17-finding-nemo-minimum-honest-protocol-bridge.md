# 2026-04-17 Finding NeMo Minimum Honest Protocol Bridge

## Question

What is the minimum honest protocol bridge between the current admitted white-box assets and the repository's localization-oriented tooling, if the goal is to advance `I-B` without pretending that `Finding NeMo` is already execution-ready or paper-faithful?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/ROADMAP.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-10-finding-nemo-mechanism-intake.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-10-finding-nemo-protocol-reconciliation.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-10-finding-nemo-observability-smoke-contract.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/2026-04-10-finding-nemo-activation-export-adapter-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/signal-access-matrix.md`

## Review

### 1. The honest bridge is migrated-DDPM observability, not paper-faithful NeMo

The current admitted white-box roots are still:

- `GSA epoch300 rerun1` for attack-side admitted evidence
- `W-1 strong-v3 full-scale` for defended admitted depth

These roots do **not** support a paper-faithful `Finding NeMo` reading because:

- current admitted assets are `DDPM/CIFAR-10`, not `Stable Diffusion v1.4`
- current repo does not expose the original `cross-attention value-layer` protocol surface
- the prompt-conditioned memorization assets required by the paper are absent

So the honest bridge cannot be:

- `Finding NeMo reproduced on current admitted assets`

It can only be:

- `migrated DDPM observability bridge on admitted GSA assets`

### 2. The bridge already has one real CPU-safe implementation surface

The minimum bridge now exists as a bounded, read-only observability route:

- fixed admitted `assets_root`
- fixed admitted `checkpoint_root`
- fixed `sample_id` / control-sample binding
- fixed `layer_selector`
- CPU-only activation export
- fixed output artifacts:
  - `summary.json`
  - `records.jsonl`
  - `tensors/.../*.pt`

This means the bridge is no longer blocked on "does any code path exist at all?".

It is now blocked on a narrower question:

- which localization observable is honest enough to trust as the first `I-B` object

### 3. The minimum honest bridge should stay below run release and below mechanism claims

Current bridge-safe wording:

- allowed:
  - `activation-only migrated DDPM observability bridge`
  - `read-only contract-probe / activation-export adapter`
  - `zero-GPU hold`
- not allowed:
  - `mechanism evidence`
  - `localized memorization neurons`
  - `paper-faithful NeMo`
  - `new GPU question`

## Minimum Bridge Contract

The minimum honest bridge is now frozen as:

1. `asset root`
   - admitted `GSA` asset family only
2. `checkpoint root`
   - admitted target checkpoint path only
3. `signal family`
   - `activations` first
   - optional `grad_norm` only as a supporting comparator
4. `layer shape`
   - one fixed selector at a time
   - default bridge selector stays `mid_block.attentions.0.to_v`
5. `sample binding`
   - fixed admitted `target-member` vs `target-nonmember` pair
6. `execution posture`
   - CPU-only
   - read-only
   - no scheduler dependency
7. `claim boundary`
   - observability exists or not
   - not mechanism proven

## Verdict

- `finding_nemo_minimum_honest_protocol_bridge_verdict = positive but bounded`

More precise reading:

1. `I-B.1` is now satisfied:
   - the minimum honest protocol bridge is `activation-only migrated DDPM observability on admitted GSA assets`
2. this bridge is already real enough to count as a repository truth object:
   - it has a fixed asset root, fixed checkpoint root, fixed selector discipline, fixed sample binding, and fixed artifact shape
3. this bridge still remains below release:
   - `gpu_release = none`
   - `queue_state = not-requestable`
4. the next honest `I-B` question is no longer “is any bridge possible?” but:
   - `which bounded localization observable is worth trusting first?`

## Next Step

- `next_live_cpu_first_lane = I-B.2 bounded localization observable selection`
- `next_gpu_candidate = none`
- `carry_forward_cpu_sidecar = higher-layer PIA provenance / I-A boundary maintenance`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: light sync required
- `workspaces/white-box/plan.md`: update required
- `Leader/materials`: no immediate sync; this is still below release-facing wording
- `Platform/Runtime`: no direct handoff; no consumer should read this as a released feature
