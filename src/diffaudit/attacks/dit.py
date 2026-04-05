"""Helpers for running the official DiT sampling code."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


DIT_WORKSPACE_FILES = (
    "sample.py",
    "download.py",
    "models.py",
)
DIT_AUTODOWNLOAD_MODELS = {"DiT-XL/2"}
DIT_AUTODOWNLOAD_IMAGE_SIZES = {256, 512}


def validate_dit_workspace(repo_root: str | Path) -> dict[str, str]:
    repo_path = Path(repo_root)
    missing = [name for name in DIT_WORKSPACE_FILES if not (repo_path / name).exists()]
    if missing or not (repo_path / "diffusion").exists():
        missing_labels = list(missing)
        if not (repo_path / "diffusion").exists():
            missing_labels.append("diffusion/")
        raise FileNotFoundError(
            f"DiT workspace is missing required files: {', '.join(missing_labels)}"
        )
    return {
        "status": "ready",
        "repo_root": str(repo_path),
        "entrypoint": str(repo_path / "sample.py"),
    }


def probe_dit_assets(
    repo_root: str | Path,
    ckpt: str | Path | None,
    model: str,
    image_size: int,
) -> dict[str, object]:
    workspace = validate_dit_workspace(repo_root)
    ckpt_path = Path(ckpt) if ckpt else None
    checks = {
        "workspace_files": True,
        "ckpt": ckpt_path.exists() if ckpt_path else True,
        "autodownload_allowed": (
            ckpt_path is not None
            or (model in DIT_AUTODOWNLOAD_MODELS and int(image_size) in DIT_AUTODOWNLOAD_IMAGE_SIZES)
        ),
    }
    missing_keys = [name for name, ready in checks.items() if not ready]
    labels = {
        "workspace_files": "DiT workspace files",
        "ckpt": "local DiT checkpoint",
        "autodownload_allowed": "official DiT autodownload constraints",
    }
    paths = {
        "repo_root": str(Path(repo_root)),
        "entrypoint": workspace["entrypoint"],
        "ckpt": str(ckpt_path) if ckpt_path else None,
        "model": model,
        "image_size": int(image_size),
    }
    missing = [
        paths["ckpt"] if key == "ckpt" and paths["ckpt"] else labels[key]
        for key in missing_keys
    ]
    return {
        "status": "ready" if all(checks.values()) else "blocked",
        "checks": checks,
        "paths": paths,
        "missing_keys": missing_keys,
        "missing": missing,
        "missing_items": [Path(item).name if Path(item).name else item for item in missing],
        "missing_description": " / ".join(labels[name] for name in missing_keys),
    }


def run_dit_sample_smoke(
    workspace: str | Path,
    repo_root: str | Path = "external/DiT",
    model: str = "DiT-XL/2",
    image_size: int = 256,
    num_sampling_steps: int = 2,
    seed: int = 0,
    ckpt: str | Path | None = None,
) -> dict[str, object]:
    asset_probe = probe_dit_assets(
        repo_root=repo_root,
        ckpt=ckpt,
        model=model,
        image_size=image_size,
    )
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    if asset_probe["status"] != "ready":
        result = {
            "status": "blocked",
            "track": "black-box",
            "method": "dit",
            "paper": "Scalable_Diffusion_Transformers_2022",
            "mode": "sample-smoke",
            "workspace": str(workspace_path),
            "checks": {
                "asset_probe_ready": False,
            },
            "asset_probe": asset_probe,
            "notes": [
                "DiT sample smoke requires the official DiT repo plus either a local checkpoint or a compatible official autodownload combination.",
            ],
        }
        (workspace_path / "summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return result

    repo_path = Path(repo_root).resolve()
    command = [
        sys.executable,
        str((repo_path / "sample.py").resolve()),
        "--model",
        model,
        "--image-size",
        str(int(image_size)),
        "--num-sampling-steps",
        str(int(num_sampling_steps)),
        "--seed",
        str(int(seed)),
    ]
    if ckpt:
        command.extend(["--ckpt", str(Path(ckpt).resolve())])

    repo_sample = repo_path / "sample.png"
    if repo_sample.exists():
        repo_sample.unlink()
    completed = subprocess.run(
        command,
        cwd=str(repo_path),
        capture_output=True,
        text=True,
        check=False,
    )
    sample_image = workspace_path / "sample.png"
    if repo_sample.exists():
        shutil.move(str(repo_sample), str(sample_image))
    checks = {
        "asset_probe_ready": True,
        "sample_script_succeeded": completed.returncode == 0,
        "sample_image_created": sample_image.exists(),
    }
    result = {
        "status": "ready" if all(checks.values()) else "error",
        "track": "black-box",
        "method": "dit",
        "paper": "Scalable_Diffusion_Transformers_2022",
        "mode": "sample-smoke",
        "workspace": str(workspace_path),
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
            "sample_image": str(sample_image),
        },
        "checks": checks,
        "runtime": {
            "repo_root": str(repo_path),
            "model": model,
            "image_size": int(image_size),
            "num_sampling_steps": int(num_sampling_steps),
            "seed": int(seed),
            "ckpt": str(Path(ckpt).resolve()) if ckpt else None,
        },
        "stdout_tail": completed.stdout.strip().splitlines()[-5:] if completed.stdout.strip() else [],
        "notes": [
            "DiT sample smoke runs the official facebookresearch/DiT sample script directly.",
            "This path validates sampling execution; it does not integrate DiT into the text-to-image recon LoRA pipeline.",
        ],
    }
    if completed.stderr.strip():
        result["stderr_tail"] = completed.stderr.strip().splitlines()[-5:]
    (workspace_path / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return result
