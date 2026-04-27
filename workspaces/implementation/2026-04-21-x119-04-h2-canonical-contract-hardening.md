# 2026-04-21 X-119 04-H2 Canonical Contract Hardening

## Question

Given that `04-H2 privacy-aware adapter` is now `prototype-implemented / contract-incomplete`, what is the minimum honest canonical `diffaudit` contract required before it can be discussed as an executable successor lane?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-21-x118-04-h2-executable-surface-reality-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/runs/h2-smp-lora-contract-smoke-20260421-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/defenses/lora_ddpm.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/defenses/smp_lora.py`
- `<DIFFAUDIT_ROOT>/Research/scripts/train_smp_lora.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_lora_smoke.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_smp_lora_runtime_tuning.py`
- `<DIFFAUDIT_ROOT>/Research/docs/report-bundles/gpt54/round2-results/04.md`

## Findings

### 1. Current repo truth

`H2` already has a real research-only prototype surface:

- LoRA injection for `DDPM UNet2DModel`
- `SMPLoRATrainer` min-max train loop
- standalone training script on image directories
- smoke and runtime-tuning tests
- one bounded CPU smoke that emits checkpoint and log artifacts

That is enough to reject the old `fallback wording only` reading.

### 2. Why it is still not a canonical `diffaudit` contract

The current surface is still a standalone prototype, not a `diffaudit` execution lane:

- no dedicated `diffaudit` CLI entry
- no asset probe that validates:
  - target checkpoint identity
  - member/nonmember roots
  - image count and image-shape compatibility
  - whether the packet is tied to current admitted assets or just ad hoc local folders
- no preparation step that freezes:
  - packet identity
  - output workspace
  - rank / lambda / optimizer / step budget
  - provenance / contract-stage fields
- no workspace-root `summary.json` emitted directly by the training path
- no attack-side review board comparing baseline vs defended checkpoint on the same packet
- no mandatory low-FPR readout
- no defense-aware attacker review boundary

### 3. Minimum honest canonical chain

Before any GPU release discussion, `H2` should be frozen to exactly four stages:

1. `probe-h2-assets`
   - verify checkpoint source
   - verify member/nonmember roots
   - verify image compatibility and bounded packet size
   - emit machine-readable readiness / blocker reason
2. `prepare-h2-contract`
   - freeze workspace path
   - freeze packet identity
   - freeze runtime hyperparameters
   - write manifest-level provenance and contract fields
3. `run-h2-defense-pilot`
   - execute bounded adapter training
   - emit workspace-root `summary.json`
   - surface artifact paths for config, log, weights, and checkpoint metadata
4. `review-h2-defense-pilot`
   - compare baseline vs defended checkpoint on the same packet
   - emit mandatory `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR`
   - explicitly mark whether the current read is transfer-only or defense-aware

## Frozen Minimum Output Contract

The first honest `run-h2-defense-pilot` packet must emit a workspace-root `summary.json` with at least:

- `status`
- `track = defense`
- `method = privacy-aware-adapter`
- `contract_stage`
- `provenance_status`
- `baseline_checkpoint`
- `defended_checkpoint`
- `packet_identity`
- `runtime`
- `artifact_paths`
- `gpu_release`
- `admitted_change`

The first honest `review-h2-defense-pilot` packet must add:

- `review_surface`
- `baseline_metrics`
- `defended_metrics`
- `metric_deltas`
- `attacker_mode`
- `low_fpr_read_order`

## Verdict

`positive but bounded`.

`X-119` freezes an honest rule:

- `H2` is real enough to deserve a canonical contract
- but not real enough to bypass contract hardening
- therefore it remains `prototype-implemented / contract-incomplete`
- and it remains below:
  - execution-ready
  - `next_gpu_candidate`
  - defense-positive wording

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `current task closed = X-119 04-H2 canonical contract hardening`
- `next CPU-first lane = I-A higher-layer boundary maintenance`
- `CPU sidecar = I-A higher-layer boundary maintenance`

## Handoff

- `Research/ROADMAP.md`: yes
- `docs/comprehensive-progress.md`: yes
- `docs/mainline-narrative.md`: yes
- `docs/reproduction-status.md`: yes
- `workspaces/implementation/challenger-queue.md`: yes
- `Platform/Runtime`: no implementation handoff yet

Reason:

This changes research-side control truth only. It still does not justify any Runtime endpoint, runner capability, or Platform schema change.
