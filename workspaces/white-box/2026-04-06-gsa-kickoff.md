# White-Box GSA Kickoff

## Status Panel

- `owner`: `codex white-box worker`
- `updated_at`: `2026-04-06 23:41:39 +08:00`
- `selected_mainline`: `2025 PoPETS White-box Membership Inference Attacks against Diffusion Models (GSA)`
- `current_state`: `repo-ready + gradient-smoke`
- `gpu_usage`: `not requested; all validation forced to CPU`
- `evidence_level`: `gradient-smoke`

## A. Current White-Box Inventory

### Papers already indexed in `Research`

- Main white-box paper: [2025-popets-white-box-membership-inference-diffusion-models-report.md](../../docs/paper-reports/white-box/2025-popets-white-box-membership-inference-diffusion-models-report.md)
- Secondary white-box route: [2024-neurips-finding-nemo-localizing-memorization-neurons-diffusion-models-report.md](../../docs/paper-reports/white-box/2024-neurips-finding-nemo-localizing-memorization-neurons-diffusion-models-report.md)
- Existing white-box workspace summary: [plan.md](plan.md)
- Existing signal framing: [signal-access-matrix.md](signal-access-matrix.md)

### Code currently available

- New local clone of the official GSA repo: [external/GSA/README.md](external/GSA/README.md)
  - clone commit: `494ce8c1a304afbb97abae59b42c65b6c5d932c2`
  - usable branches inside repo:
    - `DDPM/` for unconditional image experiments
    - `Imagen/` for text-image experiments
- Existing unified system code has no white-box adapter yet, but current shape is visible in:
  - [cli.py](../../src/diffaudit/cli.py)
  - [app.py](../../src/diffaudit/local_api/app.py)

### Checkpoint / gradient / activation conditions found in `Research`

- Paper-aligned white-box checkpoints: `not found`
- Paper-aligned white-box member/non-member dataset roots: `not found`
- Precomputed white-box gradients: `not found`
- Precomputed white-box activations: `not found`
- Existing internal-access assets in repo:
  - `external/DiT/pretrained_models/DiT-XL-2-256x256.pt`
    - real checkpoint exists, but it is a DiT sampling asset, not a paper-aligned GSA or NeMo reproduction asset
  - `D:/Code/DiffAudit/Download/black-box/supplementary/recon-assets/**/pytorch_lora_weights.safetensors`
    - black-box runtime assets, not usable as white-box gradient baselines for GSA
- White-box access status today:
  - `checkpoint`: only via newly downloaded upstream code path, not via ready experiment assets
  - `gradient`: can now be generated locally through GSA `DDPM/gen_l2_gradients_DDPM.py`
  - `activation`: no hook pipeline in repo yet
  - `optimizer_state`: not present

## B. Chosen Mainline

Chosen paper: `White-box Membership Inference Attacks against Diffusion Models`

Reason:

- It is the clearest white-box mainline in the current workspace.
- It has an official code repo, unlike `Finding NeMo`, which is heavier on prompt mining, hook logic, and SD-specific memorized prompt assets.
- Its `DDPM` path can be smoke-tested without first reconstructing a full Stable Diffusion memorization setup.
- Its core unique signal is exactly what the white-box workspace already identifies as first priority: `gradient`.

Not selected as the first executable white-box line:

- `Finding NeMo`
  - better as second-phase extension after the repository has a stable white-box internal-signal interface
  - still blocked by memorized prompt assets, SD-specific activation hooks, and causal neuron ablation flow
- `SecMI` / `PIA`
  - executable in repo, but they are tracked in this project as gray-box rather than white-box mainlines

## C. What Was Downloaded

- Official GSA repo cloned into white-box scope:
  - [external/GSA](external/GSA)
- Minimal smoke dataset created in white-box scope:
  - [sample.ppm](smoke-ddpm/train/member/sample.ppm)

No GPU assets were requested or modified.

## D. Smoke / Environment Validation

### Environment check

- `diffaudit-research` imports needed by `GSA/DDPM` are available:
  - `torch=2.5.1+cu121`
  - `torchvision=0.20.1+cu121`
  - `diffusers=0.37.1`
  - `accelerate=1.13.0`
  - `datasets=4.8.4`
- Missing dependency for the repo's final classifier script:
  - `xgboost`

### Command actually run

Run from `Research/workspaces/white-box`:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:ACCELERATE_USE_CPU='true'
conda run -n diffaudit-research python external/GSA/DDPM/gen_l2_gradients_DDPM.py `
  --train_data_dir smoke-ddpm/train `
  --resolution 32 `
  --ddpm_num_steps 20 `
  --sampling_frequency 2 `
  --attack_method 1 `
  --output_name smoke-ddpm/member-gradients.pt
```

### Result

- command exit code: `0`
- generated artifact: [member-gradients.pt](smoke-ddpm/member-gradients.pt)
- artifact stats:
  - shape: `(1, 450)`
  - dtype: `float32`
  - mean: `0.06322389096021652`
  - std: `0.2575143873691559`

Interpretation:

- The white-box DDPM gradient extraction path is runnable on this machine now.
- Current evidence is only `gradient-smoke`, not `paper-faithful reproduction`.
- The next missing layer is not gradient extraction itself, but paper-aligned assets plus a classifier dependency or replacement.

## E. White-Box-Specific Inputs For Unified System

These are the inputs the white-box line needs beyond what black-box flows usually require.

| Input | Needed for current GSA smoke | Needed for faithful GSA reproduction | Needed for NeMo-style activation line | Notes |
| --- | --- | --- | --- | --- |
| `checkpoint_path` | no | yes | yes | main white-box entry requirement |
| `shadow_checkpoint_paths[]` | no | yes | no | needed when training the attack classifier from shadow gradients |
| `dataset_root` | yes | yes | yes | current smoke uses a tiny local imagefolder |
| `member_split` / `nonmember_split` | no | yes | yes | required for real membership evaluation |
| `train_config` | no | yes | yes | includes architecture, resolution, diffusion steps, normalization, prompts/captions if conditional |
| `noise_schedule` / `timestep_sampling_spec` | yes | yes | partial | GSA depends on explicit timestep selection |
| `gradient_extraction_spec` | yes | yes | no | includes `attack_method`, `sampling_frequency`, layer reduction choice |
| `activation_hook_spec` | no | no | yes | needed for NeMo or any activation/cross-attention route |
| `prompt_or_caption_source` | no | model-dependent | yes for text-conditioned models | required once moving from unconditional DDPM to Imagen/Stable Diffusion |
| `optimizer_state` | no | optional | no | not required for baseline GSA inference-time extraction; only useful for exact training-state replay or training-dynamics audits |

Bottom line:

- For the selected GSA line, the irreducible white-box inputs are `checkpoint + dataset split + train_config + timestep/gradient extraction spec`.
- `activation` and `optimizer_state` are not first-step blockers for GSA.

## F. Minimum Shared Fields With Black-Box

White-box should reuse the same minimal envelope fields as black-box, even if the method-specific payload differs.

| Field | Meaning | Current white-box value in this kickoff |
| --- | --- | --- |
| `attack_family` | top-level track name | `white-box` |
| `method` | concrete method identifier | `gsa` |
| `workspace` | run directory containing summary/artifacts | `workspaces/white-box/smoke-ddpm` |
| `status` | execution status, not scientific maturity | `ready` for the smoke artifact, `blocked` when paper assets are absent |
| `evidence_level` | evidence maturity label shared across tracks | `gradient-smoke` |

Recommended common `evidence_level` ladder across tracks:

- `paper-read`
- `repo-ready`
- `smoke-ready`
- `gradient-smoke`
- `asset-ready`
- `experiment-ready`

For white-box, `gradient-smoke` is the first level that black-box does not need but can still fit inside the same field.

## G. If This Enters Local-API

### Minimal first `job_type`

Recommended first `job_type`:

```text
whitebox_gsa_gradient_smoke
```

Reason:

- It matches the current local API naming style of method-specific executable jobs.
- It is smaller and safer than jumping straight to a full `runtime_mainline`.
- It maps to a single command and produces a single workspace summary plus gradient artifact.

### Minimal payload shape

```json
{
  "job_type": "whitebox_gsa_gradient_smoke",
  "workspace_name": "whitebox-gsa-gradient-smoke",
  "repo_root": "workspaces/white-box/external/GSA",
  "train_data_dir": "workspaces/white-box/smoke-ddpm/train",
  "resolution": 32,
  "ddpm_num_steps": 20,
  "sampling_frequency": 2,
  "attack_method": 1,
  "device": "cpu"
}
```

Expected command shape:

```powershell
python -m diffaudit run-gsa-gradient-smoke `
  --workspace experiments/whitebox-gsa-gradient-smoke `
  --repo-root workspaces/white-box/external/GSA `
  --train-data-dir workspaces/white-box/smoke-ddpm/train `
  --resolution 32 `
  --ddpm-num-steps 20 `
  --sampling-frequency 2 `
  --attack-method 1 `
  --device cpu
```

After that, the next larger job can be:

```text
whitebox_gsa_runtime_mainline
```

but only after paper-aligned `target/shadow member/non-member` assets exist.

## H. Shortest Next Path

1. Add or isolate `xgboost` so that [test_attack_accuracy.py](external/GSA/test_attack_accuracy.py) can be exercised without leaving the white-box scope.
2. Create four tiny white-box folders inside `workspaces/white-box`:
   - `target-member`
   - `target-nonmember`
   - `shadow-member`
   - `shadow-nonmember`
3. Use the same GSA gradient script to produce four gradient tensors, then run the repo's classifier stage once end-to-end.
4. Only after that, propose shared-code integration in `src/diffaudit/`:
   - `plan-gsa`
   - `probe-gsa-assets`
   - `run-gsa-gradient-smoke`
   - Local-API `whitebox_gsa_gradient_smoke`

Current blocker for faithful reproduction is no longer "we have nothing executable".

Current blocker is: `paper-aligned checkpoints + member/non-member assets + classifier dependency gap`.

