"""Basic environment verification for the research setup."""

from __future__ import annotations

import importlib
import json


PACKAGES = [
    "numpy",
    "pandas",
    "matplotlib",
    "torch",
    "torchvision",
    "torchaudio",
    "diffusers",
    "transformers",
    "accelerate",
]


def main() -> None:
    results = {}
    for package in PACKAGES:
        module = importlib.import_module(package)
        results[package] = getattr(module, "__version__", "unknown")

    import torch

    results["cuda_available"] = torch.cuda.is_available()
    results["cuda_device_count"] = torch.cuda.device_count()
    if torch.cuda.is_available():
        results["cuda_device_name"] = torch.cuda.get_device_name(0)

    print(json.dumps(results, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
