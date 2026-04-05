"""Planning helpers for integrating the official CLiD attack flow."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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


def load_clid_score_matrix(score_path: str | Path) -> np.ndarray:
    rows: list[list[float]] = []
    with Path(score_path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append([float(value) for value in line.split("\t")])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No numeric score rows found in {score_path}")
    return np.asarray(rows, dtype=float)


def _find_single_artifact(artifact_dir: Path, pattern: str) -> Path:
    matches = sorted(artifact_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No CLiD artifact matched pattern: {pattern}")
    if len(matches) > 1:
        raise ValueError(f"Multiple CLiD artifacts matched pattern: {pattern}")
    return matches[0]


def _extract_clid_features(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("CLiD artifact matrix must have at least two columns")
    clid_avg = -matrix[:, 1:].mean(axis=1)
    return np.column_stack((matrix[:, 0], clid_avg))


def _robust_fit(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(features, axis=0)
    q1 = np.percentile(features, 25, axis=0)
    q3 = np.percentile(features, 75, axis=0)
    scale = q3 - q1
    scale[scale == 0] = 1.0
    return center, scale


def _robust_transform(features: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (features - center) / scale


def _weighted_score(features: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * features[:, 0] + alpha * features[:, 1]


def _auc_member_low(train_scores: np.ndarray, test_scores: np.ndarray) -> float:
    wins = (train_scores[:, None] < test_scores[None, :]).astype(float)
    ties = (train_scores[:, None] == test_scores[None, :]).astype(float) * 0.5
    return float((wins + ties).mean())


def _search_threshold(train_scores: np.ndarray, test_scores: np.ndarray) -> dict[str, float]:
    min_value = float(min(train_scores.min(), test_scores.min()))
    max_value = float(max(train_scores.max(), test_scores.max()))
    if min_value == max_value:
        threshold = min_value
        asr = 0.5
        auc = _auc_member_low(train_scores, test_scores)
        return {
            "threshold": threshold,
            "asr": asr,
            "auc": auc,
            "tpr_at_1pct_fpr": 0.0,
        }

    best_threshold = min_value
    best_asr = -1.0
    thresholds = np.linspace(min_value, max_value, num=2000, endpoint=True)
    tpr_points: list[float] = []
    fpr_points: list[float] = []
    for threshold in thresholds:
        tp = float((train_scores <= threshold).sum())
        tn = float((test_scores > threshold).sum())
        fp = float((test_scores <= threshold).sum())
        fn = float((train_scores > threshold).sum())
        tpr = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        asr = (tp + tn) / (tp + tn + fp + fn)
        tpr_points.append(tpr)
        fpr_points.append(fpr)
        if asr > best_asr:
            best_asr = asr
            best_threshold = float(threshold)

    order = np.argsort(fpr_points)
    sorted_fpr = np.asarray(fpr_points)[order]
    sorted_tpr = np.asarray(tpr_points)[order]
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    auc = float(integrate(sorted_tpr, sorted_fpr))
    tpr_at_1pct_fpr = 0.0
    for fpr, tpr in zip(sorted_fpr, sorted_tpr):
        if fpr >= 0.01:
            tpr_at_1pct_fpr = float(tpr)
            break

    return {
        "threshold": best_threshold,
        "asr": float(best_asr),
        "auc": auc,
        "tpr_at_1pct_fpr": tpr_at_1pct_fpr,
    }


def summarize_clid_artifacts(
    artifact_dir: str | Path,
    workspace: str | Path,
) -> dict[str, object]:
    artifact_path = Path(artifact_dir)
    shadow_train_path = _find_single_artifact(artifact_path, "*split1*TRTE_train*.txt")
    shadow_test_path = _find_single_artifact(artifact_path, "*split1*TRTE_test*.txt")
    target_train_path = _find_single_artifact(artifact_path, "*ori*TRTE_train*.txt")
    target_test_path = _find_single_artifact(artifact_path, "*ori*TRTE_test*.txt")

    shadow_train_features = _extract_clid_features(load_clid_score_matrix(shadow_train_path))
    shadow_test_features = _extract_clid_features(load_clid_score_matrix(shadow_test_path))
    target_train_features = _extract_clid_features(load_clid_score_matrix(target_train_path))
    target_test_features = _extract_clid_features(load_clid_score_matrix(target_test_path))

    center, scale = _robust_fit(np.vstack((shadow_train_features, shadow_test_features)))
    shadow_train_scaled = _robust_transform(shadow_train_features, center, scale)
    shadow_test_scaled = _robust_transform(shadow_test_features, center, scale)
    target_train_scaled = _robust_transform(target_train_features, center, scale)
    target_test_scaled = _robust_transform(target_test_features, center, scale)

    best_alpha = 0.0
    best_shadow_auc = -1.0
    best_shadow = None
    best_target = None
    for alpha in np.linspace(0.0, 1.0, num=11):
        shadow_train_scores = _weighted_score(shadow_train_scaled, float(alpha))
        shadow_test_scores = _weighted_score(shadow_test_scaled, float(alpha))
        shadow_metrics = _search_threshold(shadow_train_scores, shadow_test_scores)

        target_train_scores = _weighted_score(target_train_scaled, float(alpha))
        target_test_scores = _weighted_score(target_test_scaled, float(alpha))
        target_metrics = _search_threshold(target_train_scores, target_test_scores)

        # evaluate target split with the shadow-derived threshold
        threshold = shadow_metrics["threshold"]
        tp = float((target_train_scores <= threshold).sum())
        tn = float((target_test_scores > threshold).sum())
        fp = float((target_test_scores <= threshold).sum())
        fn = float((target_train_scores > threshold).sum())
        target_metrics["asr_via_shadow_threshold"] = (tp + tn) / (tp + tn + fp + fn)

        if shadow_metrics["auc"] > best_shadow_auc:
            best_shadow_auc = shadow_metrics["auc"]
            best_alpha = float(alpha)
            best_shadow = shadow_metrics
            best_target = target_metrics

    assert best_shadow is not None
    assert best_target is not None

    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "ready",
        "track": "black-box",
        "method": "clid",
        "paper": "CLiD_NeurIPS2024",
        "mode": "artifact-summary",
        "device": "cpu",
        "workspace": str(workspace_path),
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
            "shadow_train": str(shadow_train_path),
            "shadow_test": str(shadow_test_path),
            "target_train": str(target_train_path),
            "target_test": str(target_test_path),
        },
        "assets": {
            "artifact_dir": str(artifact_path),
            "shadow_samples": int(shadow_train_features.shape[0] + shadow_test_features.shape[0]),
            "target_samples": int(target_train_features.shape[0] + target_test_features.shape[0]),
        },
        "checks": {
            "shadow_artifacts_loaded": True,
            "target_artifacts_loaded": True,
            "feature_columns_extracted": True,
            "shadow_threshold_search": True,
            "target_replay_complete": True,
        },
        "metrics": {
            "shadow": {
                "best_alpha": round(best_alpha, 3),
                "auc": round(float(best_shadow["auc"]), 6),
                "asr": round(float(best_shadow["asr"]), 6),
                "threshold": round(float(best_shadow["threshold"]), 6),
            },
            "target": {
                "auc": round(float(best_target["auc"]), 6),
                "best_asr": round(float(best_target["asr"]), 6),
                "asr_via_shadow_threshold": round(float(best_target["asr_via_shadow_threshold"]), 6),
                "tpr_at_1pct_fpr": round(float(best_target["tpr_at_1pct_fpr"]), 6),
                "threshold": round(float(best_target["threshold"]), 6),
            },
        },
        "notes": [
            "Summary is recomputed from upstream CLiD inter_output artifacts.",
            "Current implementation reproduces the threshold-based alpha sweep, not the XGBoost path.",
        ],
    }
    (workspace_path / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return result
