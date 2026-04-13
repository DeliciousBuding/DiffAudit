# 2026-04-08 White-Box Follow-Up: DPDM W-1 Smoke Training

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-08 02:24:20 +08:00`
- `selected_defense`: `W-1 = DPDM`
- `current_state`: `smoke training completed one CUDA epoch and saved final checkpoint`
- `gpu_usage`: `single GPU`
- `evidence_level`: `runtime-smoke`

## A. What Was Fixed Before Running

- added `utils/opacus_compat.py` so local installs of `opacus` can replace the missing `src.opacus` layout
- changed StyleGAN-derived fused ops to fall back to native PyTorch when CUDA extensions cannot be built on this machine
- fixed `main.py` so child process failures now propagate as non-zero exits
- aligned `train_dpdm_base.py` with current config and `opacus` APIs:
  - `optim.name` fallback
  - `BatchMemoryManager` current signature
  - robust label normalization from one-hot to class index
- removed `Opacus`-incompatible inplace additions in `model/layerspp.py`
- updated the custom `NIN` grad sampler to accept list-based hook inputs from current `opacus`

## B. Runs And Current Outcome

Observed sequence:

1. `v2-v6` progressively exposed compatibility issues and were used to confirm each root cause
2. `v7` is the first run that entered sustained CUDA training iterations with loss logs and saved a final checkpoint

Current active smoke command:

```powershell
conda run -n diffaudit-research python D:\Code\DiffAudit\Research\external\DPDM\main.py `
  --mode train `
  --workdir runs/dpdm-cifar10-32-eps10-gpu-smoke-v7 `
  --config D:\Code\DiffAudit\Research\external\DPDM\configs\cifar10_32\train_eps_10.0.yaml `
  --root_folder D:\Code\DiffAudit\Research\external\DPDM `
  -- setup.n_gpus_per_node=1 `
     setup.omp_n_threads=8 `
     setup.backend=gloo `
     setup.master_address=127.0.0.1 `
     setup.master_port=6028 `
     data.path=D:\Code\DiffAudit\Research\external\DPDM\data\processed\cifar10.zip `
     data.dataset_params.use_labels=true `
     data.dataset_params.xflip=false `
     train.batch_size=64 `
     train.n_epochs=1 `
     train.log_freq=10 `
     train.snapshot_freq=100000 `
     train.save_freq=100000 `
     train.fid_freq=100000 `
     train.fid_threshold=100000 `
     train.save_threshold=100000 `
     train.snapshot_threshold=100000 `
     sampler.snapshot_batch_size=8 `
     sampler.fid_batch_size=8 `
     dp.max_physical_batch_size=64 `
     loss.n_classes=10 `
     loss.n_noise_samples=1
```

Latest observed training logs:

- `Loss: 0.8601, step: 10`
- `Loss: 0.8479, step: 20`
- `Loss: 0.7793, step: 30`
- `Loss: 0.7723, step: 40`
- `Loss: 0.7555, step: 50`
- `Loss: 0.7482, step: 60`
- `Loss: 0.6974, step: 70`
- `Loss: 0.6500, step: 80`
- `Loss: 0.3003, step: 1010`
- `Loss: 0.3076, step: 1060`
- `Loss: 0.3198, step: 1070`
- `Loss: 0.3157, step: 1080`
- `Loss: 0.3374, step: 1090`
- `Loss: 0.3533, step: 1100`
- `Eps-value after 1 epochs: 10.0025`
- `Saving final checkpoint.`

Checkpoint artifact:

- `runs/dpdm-cifar10-32-eps10-gpu-smoke-v7/checkpoints/final_checkpoint.pth`

## C. What This Proves

- `W-1` is no longer blocked at environment or import time
- the DPDM path now reaches real forward and backward CUDA iterations
- the current machine can run a white-box defense training smoke without requiring local CUDA toolkit compilation
- a first smoke checkpoint can now be fed into the next white-box defense evaluation stage

## D. What This Does Not Yet Prove

- a paper-aligned `W-1` baseline
- a final defense-vs-attack comparison

The current run intentionally uses `loss.n_noise_samples=1` to minimize compatibility noise while proving the full training chain.

## E. Next Step

1. build a checkpoint bridge from `DPDM final_checkpoint.pth` to the `GSA` DDPM `checkpoint-*` directory format
2. restore stronger training settings incrementally, starting with `loss.n_noise_samples`
3. promote this run from `runtime-smoke` to a first `W-1 baseline` candidate only after attack-side comparison exists

## F. Current Technical Blocker

The current `DPDM` checkpoint is:

- a single-file PyTorch payload
- keys: `model`, `ema`, `optimizer`, `step`

The current `GSA` DDPM gradient extractor consumes:

- `accelerate`-style `checkpoint-*` directories
- with files such as:
  - `model.safetensors`
  - `optimizer.bin`
  - `scheduler.bin`

So the current white-box defense blocker is:

- not missing training
- but missing checkpoint-format bridging between `DPDM` and the `GSA` evaluation path
