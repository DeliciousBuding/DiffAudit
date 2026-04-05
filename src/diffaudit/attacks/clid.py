"""Planning helpers for integrating the official CLiD attack flow."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from diffaudit.config import (
    AssetConfig,
    AttackConfig,
    AuditConfig,
    ReportConfig,
    TaskConfig,
)


CLID_ENTRYPOINTS = {
    "clid_impt": "mia_CLiD_impt.py",
    "clid_clip": "mia_CLiD_clip.py",
    "pfami": "mia_pfami.py",
}

COMMON_CLID_WORKSPACE_FILES = (
    "cal_clid_th.py",
    "cal_clid_xgb.py",
    "train_text_to_image.py",
)

CLID_MODEL_SUBDIRS = (
    "vae",
    "tokenizer",
    "text_encoder",
    "unet",
    "scheduler",
)


@dataclass(frozen=True)
class ClidPlan:
    entrypoint: str
    attack_variant: str
    dataset_train_ref: str
    dataset_test_ref: str
    model_dir: str
    max_n_samples: int
    resolution: int
    image_column: str
    caption_column: str


def build_clid_plan(config: AuditConfig) -> ClidPlan:
    if config.attack.method != "clid":
        raise ValueError(f"Unsupported attack method for CLiD plan: {config.attack.method}")
    if config.task.access_level not in {"black_box", "semi_white_box"}:
        raise ValueError("CLiD requires task.access_level = black_box or semi_white_box")

    dataset_train_ref = config.assets.dataset_train_ref
    dataset_test_ref = config.assets.dataset_test_ref
    model_dir = config.assets.model_dir
    params = config.attack.parameters
    variant = str(params.get("variant", "clid_impt")).strip()

    if variant not in CLID_ENTRYPOINTS:
        raise ValueError(f"Unsupported CLiD variant: {variant}")
    if not dataset_train_ref:
        raise ValueError("CLiD requires assets.dataset_train_ref")
    if not dataset_test_ref:
        raise ValueError("CLiD requires assets.dataset_test_ref")
    if not model_dir:
        raise ValueError("CLiD requires assets.model_dir")

    max_n_samples = int(params.get("max_n_samples", 3))
    if max_n_samples <= 0:
        raise ValueError("CLiD requires attack.parameters.max_n_samples > 0")

    return ClidPlan(
        entrypoint=CLID_ENTRYPOINTS[variant],
        attack_variant=variant,
        dataset_train_ref=dataset_train_ref,
        dataset_test_ref=dataset_test_ref,
        model_dir=model_dir,
        max_n_samples=max_n_samples,
        resolution=int(params.get("resolution", 512)),
        image_column=str(params.get("image_column", "image")),
        caption_column=str(params.get("caption_column", "text")),
    )


def validate_clid_workspace(workspace_dir: str | Path, entrypoint: str) -> dict[str, str]:
    workspace_path = Path(workspace_dir)
    required_files = (entrypoint, *COMMON_CLID_WORKSPACE_FILES)
    missing = [
        relative_path
        for relative_path in required_files
        if not (workspace_path / relative_path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"CLiD workspace is missing required files: {', '.join(missing)}"
        )

    return {
        "status": "ready",
        "workspace_dir": str(workspace_path),
        "entrypoint": str(workspace_path / entrypoint),
    }


def probe_clid_assets(
    dataset_train_ref: str,
    dataset_test_ref: str,
    model_dir: str | Path,
) -> dict[str, object]:
    model_dir_path = Path(model_dir)
    missing_subdirs = [
        subdir
        for subdir in CLID_MODEL_SUBDIRS
        if not (model_dir_path / subdir).exists()
    ]
    checks = {
        "dataset_train_ref": bool(dataset_train_ref),
        "dataset_test_ref": bool(dataset_test_ref),
        "model_dir": model_dir_path.exists(),
        "model_subdirs": not missing_subdirs,
    }
    status = "ready" if all(checks.values()) else "blocked"
    paths = {
        "model_dir": str(model_dir_path),
        "required_subdirs": [str(model_dir_path / subdir) for subdir in CLID_MODEL_SUBDIRS],
        "dataset_train_ref": dataset_train_ref,
        "dataset_test_ref": dataset_test_ref,
    }
    labels = {
        "dataset_train_ref": "dataset_train_ref",
        "dataset_test_ref": "dataset_test_ref",
        "model_dir": "model_dir",
        "model_subdirs": "model subdirs",
    }
    missing_keys = [name for name, is_ready in checks.items() if not is_ready]
    missing = [
        (
            ", ".join(missing_subdirs)
            if name == "model_subdirs"
            else str(paths[name])
        )
        for name in missing_keys
    ]
    return {
        "status": status,
        "checks": checks,
        "paths": paths,
        "missing": missing,
        "missing_keys": missing_keys,
        "missing_items": [item if item else labels[key] for item, key in zip(missing, missing_keys)],
        "missing_description": " / ".join(labels[name] for name in missing_keys),
    }


def explain_clid_assets(config: AuditConfig) -> dict[str, object]:
    plan = build_clid_plan(config)
    summary = probe_clid_assets(
        dataset_train_ref=plan.dataset_train_ref,
        dataset_test_ref=plan.dataset_test_ref,
        model_dir=plan.model_dir,
    )
    return {
        **summary,
        "attack_variant": plan.attack_variant,
        "entrypoint": plan.entrypoint,
        "max_n_samples": plan.max_n_samples,
    }


def probe_clid_dry_run(
    config: AuditConfig,
    repo_root: str | Path,
) -> tuple[int, dict[str, object]]:
    try:
        plan = build_clid_plan(config)
        workspace = validate_clid_workspace(repo_root, plan.entrypoint)
        summary = probe_clid_assets(
            dataset_train_ref=plan.dataset_train_ref,
            dataset_test_ref=plan.dataset_test_ref,
            model_dir=plan.model_dir,
        )
        if summary["status"] != "ready":
            return 1, {
                "status": "blocked",
                "repo_root": str(repo_root),
                **summary,
            }
        entrypoint_path = Path(repo_root) / plan.entrypoint
        entrypoint_text = entrypoint_path.read_text(encoding="utf-8")
    except (FileNotFoundError, ValueError) as exc:
        return 1, {
            "status": "blocked",
            "error": str(exc),
            "repo_root": str(repo_root),
        }
    except Exception as exc:
        return 2, {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "repo_root": str(repo_root),
        }

    checks = {
        **summary["checks"],
        "workspace_files": True,
        "entrypoint_has_attack_flag": "flags.attack" in entrypoint_text,
        "script_uses_load_dataset": "load_dataset" in entrypoint_text,
        "script_uses_diffusers": "AutoencoderKL" in entrypoint_text,
    }
    status = "ready" if all(checks.values()) else "blocked"
    missing_keys = [name for name, is_ready in checks.items() if not is_ready]
    labels = {
        "dataset_train_ref": "dataset_train_ref",
        "dataset_test_ref": "dataset_test_ref",
        "model_dir": "model_dir",
        "model_subdirs": "model subdirs",
        "workspace_files": "workspace files",
        "entrypoint_has_attack_flag": "entrypoint attack flag",
        "script_uses_load_dataset": "load_dataset import",
        "script_uses_diffusers": "diffusers import",
    }
    extra_paths = {
        "workspace_files": str(Path(repo_root)),
        "entrypoint_has_attack_flag": str(entrypoint_path),
        "script_uses_load_dataset": str(entrypoint_path),
        "script_uses_diffusers": str(entrypoint_path),
    }
    missing = list(summary["missing"])
    for key in missing_keys:
        if key not in summary["missing_keys"]:
            missing.append(extra_paths[key])

    payload = {
        "status": status,
        "repo_root": str(Path(repo_root)),
        "entrypoint": workspace["entrypoint"],
        "attack_variant": plan.attack_variant,
        "checks": checks,
        "paths": {
            **summary["paths"],
            "entrypoint": str(entrypoint_path),
        },
        "missing": missing,
        "missing_keys": missing_keys,
        "missing_items": [Path(item).name if Path(item).name else item for item in missing],
        "missing_description": " / ".join(labels[name] for name in missing_keys),
    }
    return (0 if status == "ready" else 1), payload


def run_clid_dry_run_smoke(
    workspace: str | Path,
    repo_root: str | Path = "external/CLiD",
    variant: str = "clid_impt",
) -> dict[str, object]:
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    synthetic_root = workspace_path / "synthetic-assets"
    model_dir = synthetic_root / "model"
    for subdir in CLID_MODEL_SUBDIRS:
        (model_dir / subdir).mkdir(parents=True, exist_ok=True)

    config = AuditConfig(
        task=TaskConfig(
            name="clid-dry-run-smoke",
            model_family="diffusion",
            access_level="black_box",
        ),
        assets=AssetConfig(
            dataset_id="synthetic-text-to-image-split",
            model_id="synthetic-clid-diffusers-model",
            model_dir=model_dir.as_posix(),
            dataset_train_ref="synthetic/train-ref",
            dataset_test_ref="synthetic/test-ref",
        ),
        attack=AttackConfig(
            method="clid",
            num_samples=8,
            parameters={
                "variant": variant,
                "max_n_samples": 3,
            },
        ),
        report=ReportConfig(output_dir="experiments/clid-dry-run-smoke"),
    )
    exit_code, payload = probe_clid_dry_run(config, repo_root)
    result = {
        "status": payload["status"],
        "track": "black-box",
        "method": "clid",
        "paper": "CLiD_NeurIPS2024",
        "mode": "dry-run-smoke",
        "device": "cpu",
        "workspace": str(workspace_path),
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
        },
        "assets": {
            "repo_root": str(Path(repo_root)),
            "attack_variant": variant,
            "dataset_train_ref": "synthetic/train-ref",
            "dataset_test_ref": "synthetic/test-ref",
        },
        "checks": {
            **payload.get("checks", {}),
            "dry_run_ready": exit_code == 0 and payload["status"] == "ready",
            "synthetic_assets_cleaned": True,
        },
        "notes": [
            "Dry-run smoke uses synthetic model folders to verify CLiD config and workspace readiness.",
            "Synthetic assets are removed after the summary is written.",
        ],
    }
    if "entrypoint" in payload:
        result["entrypoint"] = payload["entrypoint"]
    if exit_code != 0 and "error" in payload:
        result["error"] = payload["error"]

    (workspace_path / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    shutil.rmtree(synthetic_root, ignore_errors=True)
    return result
