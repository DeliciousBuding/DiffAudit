"""Helpers for the reconstruction-based black-box attack line."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from diffaudit.config import AuditConfig


RECON_WORKSPACE_FILES = (
    "inference.py",
    "cal_embedding.py",
    "test_accuracy.py",
    "train_text_to_image_lora.py",
)


@dataclass(frozen=True)
class ReconPlan:
    finetune_entrypoint: str
    inference_entrypoint: str
    embedding_entrypoint: str
    evaluation_entrypoint: str
    pretrained_model_name_or_path: str
    dataset_root: str
    model_dir: str
    num_validation_images: int
    inference_steps: int
    eval_method: str


def build_recon_plan(config: AuditConfig) -> ReconPlan:
    if config.attack.method != "recon":
        raise ValueError(f"Unsupported attack method for reconstruction plan: {config.attack.method}")
    if config.task.access_level != "black_box":
        raise ValueError("Reconstruction attack requires task.access_level = black_box")

    dataset_root = config.assets.dataset_root
    model_dir = config.assets.model_dir
    params = config.attack.parameters
    pretrained_model_name_or_path = str(params.get("pretrained_model_name_or_path", "")).strip()
    if not dataset_root:
        raise ValueError("Reconstruction attack requires assets.dataset_root")
    if not model_dir:
        raise ValueError("Reconstruction attack requires assets.model_dir")
    if not pretrained_model_name_or_path:
        raise ValueError("Reconstruction attack requires attack.parameters.pretrained_model_name_or_path")

    num_validation_images = int(params.get("num_validation_images", 3))
    inference_steps = int(params.get("inference_steps", 30))
    if num_validation_images <= 0:
        raise ValueError("Reconstruction attack requires attack.parameters.num_validation_images > 0")
    if inference_steps <= 0:
        raise ValueError("Reconstruction attack requires attack.parameters.inference_steps > 0")

    return ReconPlan(
        finetune_entrypoint="train_text_to_image_lora.py",
        inference_entrypoint="inference.py",
        embedding_entrypoint="cal_embedding.py",
        evaluation_entrypoint="test_accuracy.py",
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        dataset_root=dataset_root,
        model_dir=model_dir,
        num_validation_images=num_validation_images,
        inference_steps=inference_steps,
        eval_method=str(params.get("eval_method", "threshold")),
    )


def validate_recon_workspace(workspace_dir: str | Path) -> dict[str, str]:
    workspace_path = Path(workspace_dir)
    missing = [
        relative_path
        for relative_path in RECON_WORKSPACE_FILES
        if not (workspace_path / relative_path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Reconstruction attack workspace is missing required files: {', '.join(missing)}"
        )
    return {
        "status": "ready",
        "workspace_dir": str(workspace_path),
        "inference_entrypoint": str(workspace_path / "inference.py"),
    }


def probe_recon_assets(
    dataset_root: str | Path,
    model_dir: str | Path,
    pretrained_model_name_or_path: str,
) -> dict[str, object]:
    dataset_path = Path(dataset_root)
    model_path = Path(model_dir)
    checks = {
        "dataset_root": dataset_path.exists(),
        "model_dir": model_path.exists(),
        "pretrained_model_name_or_path": bool(pretrained_model_name_or_path),
    }
    status = "ready" if all(checks.values()) else "blocked"
    paths = {
        "dataset_root": str(dataset_path),
        "model_dir": str(model_path),
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
    }
    labels = {
        "dataset_root": "dataset_root",
        "model_dir": "model_dir",
        "pretrained_model_name_or_path": "pretrained model ref",
    }
    missing_keys = [name for name, is_ready in checks.items() if not is_ready]
    missing = [paths[name] or labels[name] for name in missing_keys]
    return {
        "status": status,
        "checks": checks,
        "paths": paths,
        "missing": missing,
        "missing_keys": missing_keys,
        "missing_items": [Path(item).name if Path(item).name else item for item in missing],
        "missing_description": " / ".join(labels[name] for name in missing_keys),
    }


def explain_recon_assets(config: AuditConfig) -> dict[str, object]:
    plan = build_recon_plan(config)
    summary = probe_recon_assets(
        dataset_root=plan.dataset_root,
        model_dir=plan.model_dir,
        pretrained_model_name_or_path=plan.pretrained_model_name_or_path,
    )
    return {
        **summary,
        "num_validation_images": plan.num_validation_images,
        "inference_steps": plan.inference_steps,
        "eval_method": plan.eval_method,
    }


def probe_recon_dry_run(
    config: AuditConfig,
    repo_root: str | Path,
) -> tuple[int, dict[str, object]]:
    try:
        plan = build_recon_plan(config)
        workspace = validate_recon_workspace(repo_root)
        summary = probe_recon_assets(
            dataset_root=plan.dataset_root,
            model_dir=plan.model_dir,
            pretrained_model_name_or_path=plan.pretrained_model_name_or_path,
        )
        if summary["status"] != "ready":
            return 1, {
                "status": "blocked",
                "repo_root": str(repo_root),
                **summary,
            }
        inference_text = (Path(repo_root) / "inference.py").read_text(encoding="utf-8")
        embedding_text = (Path(repo_root) / "cal_embedding.py").read_text(encoding="utf-8")
        evaluation_text = (Path(repo_root) / "test_accuracy.py").read_text(encoding="utf-8")
    except (FileNotFoundError, ValueError) as exc:
        return 1, {
            "status": "blocked",
            "error": str(exc),
            "repo_root": str(repo_root),
        }

    checks = {
        **summary["checks"],
        "workspace_files": True,
        "inference_has_pipeline": "StableDiffusionPipeline" in inference_text,
        "embedding_has_feature_extractor": (
            "DeiTFeatureExtractor" in embedding_text or "AutoImageProcessor" in embedding_text
        ),
        "evaluation_has_classifier": "DefineClassifier" in evaluation_text,
    }
    status = "ready" if all(checks.values()) else "blocked"
    labels = {
        "dataset_root": "dataset_root",
        "model_dir": "model_dir",
        "pretrained_model_name_or_path": "pretrained model ref",
        "workspace_files": "workspace files",
        "inference_has_pipeline": "StableDiffusionPipeline import",
        "embedding_has_feature_extractor": "feature extractor import",
        "evaluation_has_classifier": "classifier definition",
    }
    extra_paths = {
        "workspace_files": str(Path(repo_root)),
        "inference_has_pipeline": str(Path(repo_root) / "inference.py"),
        "embedding_has_feature_extractor": str(Path(repo_root) / "cal_embedding.py"),
        "evaluation_has_classifier": str(Path(repo_root) / "test_accuracy.py"),
    }
    missing_keys = [name for name, is_ready in checks.items() if not is_ready]
    missing = list(summary["missing"])
    for key in missing_keys:
        if key not in summary["missing_keys"]:
            missing.append(extra_paths[key])

    payload = {
        "status": status,
        "repo_root": str(Path(repo_root)),
        "entrypoint": workspace["inference_entrypoint"],
        "checks": checks,
        "paths": {
            **summary["paths"],
            "inference_entrypoint": str(Path(repo_root) / "inference.py"),
            "embedding_entrypoint": str(Path(repo_root) / "cal_embedding.py"),
            "evaluation_entrypoint": str(Path(repo_root) / "test_accuracy.py"),
        },
        "missing": missing,
        "missing_keys": missing_keys,
        "missing_items": [Path(item).name if Path(item).name else item for item in missing],
        "missing_description": " / ".join(labels[name] for name in missing_keys),
    }
    return (0 if status == "ready" else 1), payload


def _extract_recon_scores(score_file: str | Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        payload = torch.load(score_file, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(score_file, map_location="cpu")
    features = []
    labels = []
    for item in payload:
        feature = item[0]
        label = int(item[1])
        if isinstance(feature, list):
            features.append(float(feature[0]))
        else:
            features.append(float(feature))
        labels.append(label)
    return np.asarray(features, dtype=float), np.asarray(labels, dtype=int)


def _threshold_metrics(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> dict[str, float]:
    min_value = float(min(member_scores.min(), nonmember_scores.min()))
    max_value = float(max(member_scores.max(), nonmember_scores.max()))
    if min_value == max_value:
        return {
            "threshold": min_value,
            "asr": 0.5,
            "auc": 0.5,
            "tpr_at_1pct_fpr": 0.0,
        }

    thresholds = np.linspace(min_value, max_value, num=1000, endpoint=True)
    best_threshold = min_value
    best_asr = -1.0
    best_tpr_at_1pct = 0.0
    for threshold in thresholds:
        tp = float((member_scores > threshold).sum())
        tn = float((nonmember_scores <= threshold).sum())
        fp = float((nonmember_scores > threshold).sum())
        fn = float((member_scores <= threshold).sum())
        asr = (tp + tn) / (tp + tn + fp + fn)
        tpr = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        if asr > best_asr:
            best_asr = asr
            best_threshold = float(threshold)
        if fpr >= 0.01:
            best_tpr_at_1pct = float(tpr)
            break

    wins = (member_scores[:, None] > nonmember_scores[None, :]).astype(float)
    ties = (member_scores[:, None] == nonmember_scores[None, :]).astype(float) * 0.5
    auc = float((wins + ties).mean())
    return {
        "threshold": best_threshold,
        "asr": float(best_asr),
        "auc": auc,
        "tpr_at_1pct_fpr": best_tpr_at_1pct,
    }


def run_recon_eval_smoke(workspace: str | Path) -> dict[str, object]:
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    artifact_root = workspace_path / "synthetic-score-artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    score_payloads = {
        "target_member": [([0.90], 1), ([0.82], 1), ([0.88], 1), ([0.79], 1)],
        "target_non_member": [([0.11], 0), ([0.19], 0), ([0.23], 0), ([0.15], 0)],
        "shadow_member": [([0.86], 1), ([0.80], 1), ([0.84], 1), ([0.78], 1)],
        "shadow_non_member": [([0.17], 0), ([0.14], 0), ([0.20], 0), ([0.12], 0)],
    }
    score_paths = {}
    for name, payload in score_payloads.items():
        score_path = artifact_root / f"{name}.pt"
        torch.save(payload, score_path)
        score_paths[name] = str(score_path)

    shadow_member_scores, _ = _extract_recon_scores(score_paths["shadow_member"])
    shadow_nonmember_scores, _ = _extract_recon_scores(score_paths["shadow_non_member"])
    target_member_scores, _ = _extract_recon_scores(score_paths["target_member"])
    target_nonmember_scores, _ = _extract_recon_scores(score_paths["target_non_member"])
    shadow_metrics = _threshold_metrics(shadow_member_scores, shadow_nonmember_scores)
    target_metrics = _threshold_metrics(target_member_scores, target_nonmember_scores)

    result = {
        "status": "ready",
        "track": "black-box",
        "method": "recon",
        "paper": "BlackBox_Reconstruction_ArXiv2023",
        "mode": "eval-smoke",
        "device": "cpu",
        "workspace": str(workspace_path),
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
        },
        "checks": {
            "synthetic_scores_created": True,
            "shadow_metrics_computed": True,
            "target_metrics_computed": True,
            "synthetic_artifacts_cleaned": True,
        },
        "metrics": {
            "shadow_auc": round(float(shadow_metrics["auc"]), 6),
            "shadow_asr": round(float(shadow_metrics["asr"]), 6),
            "target_auc": round(float(target_metrics["auc"]), 6),
            "target_asr": round(float(target_metrics["asr"]), 6),
            "tpr_at_1pct_fpr": round(float(target_metrics["tpr_at_1pct_fpr"]), 6),
            "auc": round(float(target_metrics["auc"]), 6),
        },
        "notes": [
            "Eval smoke uses synthetic score artifacts compatible with the upstream accuracy script shape.",
            "Smoke verifies artifact format and threshold-style black-box evaluation only.",
        ],
    }
    (workspace_path / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    shutil.rmtree(artifact_root, ignore_errors=True)
    return result


def summarize_recon_artifacts(
    artifact_dir: str | Path,
    workspace: str | Path,
) -> dict[str, object]:
    artifact_path = Path(artifact_dir)
    score_paths = {
        "target_member": artifact_path / "target_member.pt",
        "target_non_member": artifact_path / "target_non_member.pt",
        "shadow_member": artifact_path / "shadow_member.pt",
        "shadow_non_member": artifact_path / "shadow_non_member.pt",
    }
    for name, score_path in score_paths.items():
        if not score_path.exists():
            raise FileNotFoundError(f"Missing reconstruction artifact: {name} -> {score_path}")

    shadow_member_scores, _ = _extract_recon_scores(score_paths["shadow_member"])
    shadow_nonmember_scores, _ = _extract_recon_scores(score_paths["shadow_non_member"])
    target_member_scores, _ = _extract_recon_scores(score_paths["target_member"])
    target_nonmember_scores, _ = _extract_recon_scores(score_paths["target_non_member"])
    shadow_metrics = _threshold_metrics(shadow_member_scores, shadow_nonmember_scores)
    target_metrics = _threshold_metrics(target_member_scores, target_nonmember_scores)

    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "ready",
        "track": "black-box",
        "method": "recon",
        "paper": "BlackBox_Reconstruction_ArXiv2023",
        "mode": "artifact-summary",
        "device": "cpu",
        "workspace": str(workspace_path),
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
        },
        "assets": {
            "artifact_files": sorted(score_paths.keys()),
        },
        "checks": {
            "target_member_loaded": True,
            "target_non_member_loaded": True,
            "shadow_member_loaded": True,
            "shadow_non_member_loaded": True,
            "threshold_metrics_computed": True,
        },
        "metrics": {
            "shadow_auc": round(float(shadow_metrics["auc"]), 6),
            "shadow_asr": round(float(shadow_metrics["asr"]), 6),
            "shadow_threshold": round(float(shadow_metrics["threshold"]), 6),
            "target_auc": round(float(target_metrics["auc"]), 6),
            "target_asr": round(float(target_metrics["asr"]), 6),
            "target_threshold": round(float(target_metrics["threshold"]), 6),
            "tpr_at_1pct_fpr": round(float(target_metrics["tpr_at_1pct_fpr"]), 6),
            "auc": round(float(target_metrics["auc"]), 6),
        },
        "notes": [
            "Summary is recomputed from reconstruction score artifacts compatible with the upstream accuracy script.",
            "Current implementation reproduces the threshold-style path, not the neural classifier training loop.",
        ],
    }
    (workspace_path / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return result
