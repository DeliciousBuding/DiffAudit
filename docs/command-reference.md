# Command Reference

This page keeps runnable command recipes out of the root README. It is a starting point for local validation and bounded research runs, not a benchmark claim list.

Run commands from the `Research/` repository root unless a command says otherwise.

## Environment

Create the default environment:

```powershell
conda env create -f environment.yml
conda activate diffaudit-research
python scripts/bootstrap_research_env.py --install
python scripts/verify_env.py
python -m diffaudit --help
```

Update an existing environment:

```powershell
conda env update -f environment.yml --prune
conda activate diffaudit-research
python scripts/bootstrap_research_env.py --install
```

Use the optional newer-GPU environment only after the default stack hits a real CUDA compatibility error:

```powershell
conda env create -f environment.gpu-cu128.yml
conda activate diffaudit-research
python scripts/bootstrap_research_env.py --install
```

If the shell has not activated conda, prefix commands with:

```powershell
conda run -n diffaudit-research python scripts/verify_env.py
conda run -n diffaudit-research python -m diffaudit --help
```

## Local Asset Binding

Create the ignored local config:

```powershell
Copy-Item configs/assets/team.local.template.yaml configs/assets/team.local.yaml
```

Fill the paths in `configs/assets/team.local.yaml`, then render per-line local configs:

```powershell
python scripts/render_team_local_configs.py
```

Do not commit personal absolute paths. Shared raw assets belong under `<DIFFAUDIT_ROOT>/Download/`; ignored upstream code clones belong under `Research/external/`.

## Smoke Pipeline

Run the minimal smoke pipeline:

```powershell
python -m diffaudit run-smoke --config configs/benchmarks/secmi_smoke.yaml --workspace .
```

Run the local check wrapper:

```powershell
python scripts/run_local_checks.py
```

## Black-Box

Plan `recon`:

```powershell
python -m diffaudit plan-recon --config configs/attacks/recon_plan.yaml
python -m diffaudit probe-recon-assets --config configs/attacks/recon_plan.yaml
python -m diffaudit dry-run-recon --config configs/attacks/recon_plan.yaml --repo-root external/Reconstruction-based-Attack
```

Run `recon` smoke and artifact paths:

```powershell
python -m diffaudit run-recon-eval-smoke --workspace experiments/recon-eval-smoke
python -m diffaudit run-recon-mainline-smoke --workspace experiments/recon-mainline-smoke --repo-root external/Reconstruction-based-Attack --method threshold
python -m diffaudit probe-recon-score-artifacts --artifact-dir path/to/recon-scores
python -m diffaudit run-recon-artifact-mainline --artifact-dir path/to/recon-scores --workspace experiments/recon-artifact-mainline --repo-root external/Reconstruction-based-Attack --method threshold
```

Run the strict paper-faithful Stage 0 gate:

```powershell
python -m diffaudit check-recon-stage0-paper-gate --repo-root external/Reconstruction-based-Attack --bundle-root "$env:DIFFAUDIT_ROOT/Download/black-box/supplementary/recon-assets/ndss-2025-blackbox-membership-inference-fine-tuned-diffusion-models" --attack-scenario attack-i
```

Plan `variation`:

```powershell
python -m diffaudit plan-variation --config configs/attacks/variation_plan.yaml
python -m diffaudit probe-variation-assets --config configs/attacks/variation_plan.yaml
python -m diffaudit dry-run-variation --config configs/attacks/variation_plan.yaml
python -m diffaudit run-variation-synth-smoke --workspace experiments/variation-synth-smoke
```

Plan `CLiD`:

```powershell
python -m diffaudit plan-clid --config configs/attacks/clid_plan.yaml
python -m diffaudit probe-clid-assets --config configs/attacks/clid_plan.yaml
python -m diffaudit dry-run-clid --config configs/attacks/clid_plan.yaml --repo-root external/CLiD
python -m diffaudit run-clid-dry-run-smoke --workspace experiments/clid-dry-run-smoke --repo-root external/CLiD
python -m diffaudit summarize-clid-artifacts --artifact-dir "$env:DIFFAUDIT_ROOT/Download/black-box/supplementary/clid-mia-supplementary/contents/CLID_MIA/inter_output/CLID" --workspace experiments/clid-artifact-summary
```

## Gray-Box

Plan and probe `PIA`:

```powershell
python -m diffaudit plan-pia --config configs/attacks/pia_plan.yaml
python -m diffaudit probe-pia-assets --config configs/attacks/pia_plan.yaml --member-split-root external/PIA/DDPM
python -m diffaudit dry-run-pia --config configs/attacks/pia_plan.yaml --repo-root external/PIA --member-split-root external/PIA/DDPM
```

Run bounded `PIA` previews:

```powershell
python -m diffaudit runtime-probe-pia --config configs/attacks/pia_plan.yaml --repo-root external/PIA --member-split-root external/PIA/DDPM --device cpu
python -m diffaudit runtime-preview-pia --config configs/attacks/pia_plan.yaml --repo-root external/PIA --member-split-root external/PIA/DDPM --device cpu --preview-batch-size 4
python -m diffaudit run-pia-runtime-smoke --workspace experiments/pia-runtime-smoke-cpu --repo-root external/PIA --device cpu
python -m diffaudit run-pia-synth-smoke --workspace experiments/pia-synth-smoke-cpu --repo-root external/PIA --device cpu
```

Plan and probe `SecMI`:

```powershell
python -m diffaudit plan-secmi --config configs/attacks/secmi_plan.yaml
python -m diffaudit probe-secmi-assets --config configs/attacks/secmi_plan.yaml
python -m diffaudit prepare-secmi --config configs/attacks/secmi_plan.yaml --repo-root third_party/secmi
python -m diffaudit dry-run-secmi --config configs/attacks/secmi_plan.yaml --repo-root third_party/secmi
python -m diffaudit runtime-probe-secmi --config configs/attacks/secmi_plan.yaml --repo-root third_party/secmi
```

Bootstrap local `SecMI` smoke assets:

```powershell
python -m diffaudit bootstrap-secmi-smoke-assets --target-dir tmp/secmi-smoke-assets
```

## White-Box

Probe and run `GSA`:

```powershell
python -m diffaudit probe-gsa-assets --repo-root workspaces/white-box/external/GSA --assets-root workspaces/white-box/assets/gsa
python -m diffaudit run-gsa-runtime-mainline --workspace workspaces/white-box/runs/gsa-runtime-mainline --repo-root workspaces/white-box/external/GSA --assets-root workspaces/white-box/assets/gsa --resolution 32 --ddpm-num-steps 20 --sampling-frequency 2 --attack-method 1
```

Probe and sample `DiT`:

```powershell
python -m diffaudit probe-dit-assets --repo-root external/DiT --model "DiT-XL/2" --image-size 256
python -m diffaudit run-dit-sample-smoke --workspace experiments/dit-sample-smoke --repo-root external/DiT --model "DiT-XL/2" --image-size 256 --num-sampling-steps 2 --seed 0
```

## Runtime Boundary

The active runtime service is a sibling repository, not this `Research/` repo:

```powershell
cd ../Runtime-Server
go run ./cmd/runtime --host 127.0.0.1 --port 8765
```

Research commands may write summaries and manifests that Runtime or Platform consumes later, but service deployment and HTTP control-plane work should stay in `Runtime-Server/` or `Platform/`.
