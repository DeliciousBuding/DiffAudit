# Research Environment

This project uses a dedicated conda environment named `diffaudit-research`.

## Scope

The environment is intentionally designed for the current research stage:

- Python numerical computing: `numpy`, `pandas`, `matplotlib`
- PyTorch fundamentals: tensors, autograd, training and inference basics
- Diffusion-model research preparation: `diffusers`, `transformers`, `accelerate`
- Notebook-based reading, exploration, and small-scale experiments

It is not yet tied to a specific membership-inference implementation.

## Why Python 3.11

Python 3.11 is chosen instead of 3.12 because the current PyTorch and diffusion-model ecosystem is generally more stable on 3.11 for research workflows.

## Package Strategy

This setup uses conda only for the isolated Python runtime and uses `pip` for the scientific and deep-learning packages. That is intentional because the local conda cache currently contains corrupted scientific packages.

## Create The Environment

```powershell
conda env create -f environment.yml
conda activate diffaudit-research
python -m ipykernel install --user --name diffaudit-research --display-name "Python (diffaudit-research)"
```

GPU-enabled PyTorch is pinned to the CUDA 12.1 wheel index in `environment.yml`.
The current validated stack is:

- `torch==2.5.1+cu121`
- `torchvision==0.20.1+cu121`
- `torchaudio==2.5.1+cu121`

## Verify

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import numpy, pandas, matplotlib, diffusers, transformers"
python scripts/verify_env.py
```

## Later Additions

Only add algorithm-specific dependencies after the first concrete attack direction is confirmed. That keeps the environment stable during the survey stage.
