"""Bootstrap and verify the DiffAudit research environment."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path


REQUIRED_IMPORTS = [
    "diffaudit",
    "numpy",
    "pandas",
    "matplotlib",
    "torch",
    "torchvision",
    "torchaudio",
    "diffusers",
    "transformers",
    "accelerate",
    "xgboost",
    "fastapi",
    "uvicorn",
]


def run(cmd: list[str], cwd: Path) -> None:
    completed = subprocess.run(cmd, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def verify_imports() -> dict[str, str]:
    results: dict[str, str] = {}
    for name in REQUIRED_IMPORTS:
        module = importlib.import_module(name)
        results[name] = getattr(module, "__version__", "unknown")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--install",
        action="store_true",
        help="run `python -m pip install -e .` before verification",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    if args.install:
        run([sys.executable, "-m", "pip", "install", "-e", "."], repo_root)

    results = verify_imports()
    print(json.dumps(results, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
