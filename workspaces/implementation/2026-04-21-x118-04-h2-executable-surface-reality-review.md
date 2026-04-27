# 2026-04-21 X-118 04-H2 Executable Surface Reality Review

## Question

After `X-114` froze `04-H2 privacy-aware adapter` as fallback wording only, does the current repo still support that reading, or has `H2` already crossed into a real prototype surface that should change control-plane truth?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/docs/report-bundles/gpt54/round2-results/04.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-18-x114-04-defense-post-h1-family-review.md`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/defenses/lora_ddpm.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/defenses/smp_lora.py`
- `<DIFFAUDIT_ROOT>/Research/scripts/train_smp_lora.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_lora_smoke.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_smp_lora_runtime_tuning.py`

## Method

1. Re-read the post-`H1` family-review claim that `H2` had no implementation / CLI / test surface.
2. Inspect the current repo for actual `MP/SMP-LoRA` code paths.
3. Execute one bounded CPU smoke:

```powershell
python <DIFFAUDIT_ROOT>/Research/scripts/train_smp_lora.py `
  --random_init `
  --member_dir <DIFFAUDIT_ROOT>/Research/workspaces/implementation/runs/h2-smp-lora-contract-smoke-20260421-r1/member `
  --nonmember_dir <DIFFAUDIT_ROOT>/Research/workspaces/implementation/runs/h2-smp-lora-contract-smoke-20260421-r1/nonmember `
  --output_dir <DIFFAUDIT_ROOT>/Research/workspaces/implementation/runs/h2-smp-lora-contract-smoke-20260421-r1/output `
  --rank 4 `
  --lambda_coeff 0.05 `
  --num_epochs 1 `
  --batch_size 1 `
  --num_workers 0 `
  --device cpu `
  --save_every 999999
```

## Evidence

### 1. `H2` is not wording-only anymore

The repo now already contains:

- a LoRA injection layer for DDPM:
  - `src/diffaudit/defenses/lora_ddpm.py`
- an `SMPLoRATrainer` training loop:
  - `src/diffaudit/defenses/smp_lora.py`
- a runnable training script:
  - `scripts/train_smp_lora.py`
- smoke / runtime-tuning tests:
  - `tests/test_lora_smoke.py`
  - `tests/test_smp_lora_runtime_tuning.py`

So the old `X-114` reading "no implementation / no test surface" is no longer factually correct on the current repo tree.

### 2. One real bounded smoke succeeded

Canonical smoke artifact:

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/runs/h2-smp-lora-contract-smoke-20260421-r1/summary.json`

Key emitted artifacts:

- `output/config.json`
- `output/final/checkpoint_meta.json`
- `output/final/lora_summary.json`
- `output/final/training_log.json`
- `output/final/lora_weights.pt`
- `output/final/proxy_weights.pt`

Observed smoke facts:

- `rank = 4`
- `lambda_coeff = 0.05`
- `method = smp`
- `num_lora_layers = 12`
- `total_lora_params = 49152`
- `overall_compression_ratio = 64.0`
- one actual training step completed:
  - `adaptation_loss = 1.07795`
  - `mi_gain = 0.5`
  - `objective = 1.105477`

## Verdict

`positive but bounded`.

More precise reading:

1. `04-H2 privacy-aware adapter` is no longer only fallback wording.
2. It now has a **research-only prototype surface**:
   - prototype implementation
   - script entrypoint
   - smoke/runtime-tuning tests
   - one real bounded CPU smoke artifact
3. But it is still **not** a canonical `diffaudit` execution contract:
   - no dedicated `diffaudit` CLI entry
   - no canonical asset probe / prep / run / review chain
   - no attack-side review board on current admitted assets
   - no honest GPU release

Therefore the correct control-plane update is:

- promote `H2` from `fallback wording only`
- to `prototype-implemented / contract-incomplete`

But do **not** promote it to:

- `execution-ready successor`
- `next_gpu_candidate`
- defense-positive line

## Control Decision

- `active GPU question = none`
- `next_gpu_candidate = none`
- `current task closed = X-118 04-H2 executable surface reality review`
- `next CPU-first lane = X-119 04-H2 canonical contract hardening`
- `CPU sidecar = I-A higher-layer boundary maintenance`

## Handoff

- `Research/ROADMAP.md`: update required
- `docs/comprehensive-progress.md`: update required
- `docs/mainline-narrative.md`: update required
- `docs/reproduction-status.md`: update required
- `workspaces/implementation/challenger-queue.md`: update required
- `Platform/Runtime`: no schema change required yet

Reason:

This changes research-side control truth, but it still does not alter admitted rows, Runtime endpoints, or Platform snapshot shape.
