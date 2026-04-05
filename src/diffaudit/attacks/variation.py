"""Helpers for API-only black-box variation attacks."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from diffaudit.config import AuditConfig


SUPPORTED_DISTANCE_METRICS = {"l2", "cosine"}
SUPPORTED_THRESHOLD_STRATEGIES = {"threshold", "distribution"}
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class VariationPlan:
    entrypoint: str
    query_image_root: str
    endpoint: str
    num_averages: int
    distance_metric: str
    threshold_strategy: str


def build_variation_plan(config: AuditConfig) -> VariationPlan:
    if config.attack.method != "variation":
        raise ValueError(f"Unsupported attack method for variation plan: {config.attack.method}")
    if config.task.access_level != "black_box":
        raise ValueError("Variation attack requires task.access_level = black_box")

    query_image_root = config.assets.dataset_root
    params = config.attack.parameters
    endpoint = str(params.get("endpoint", "")).strip()
    distance_metric = str(params.get("distance_metric", "l2")).strip()
    threshold_strategy = str(params.get("threshold_strategy", "threshold")).strip()
    num_averages = int(params.get("num_averages", 3))

    if not query_image_root:
        raise ValueError("Variation attack requires assets.dataset_root")
    if not endpoint:
        raise ValueError("Variation attack requires attack.parameters.endpoint")
    if num_averages <= 0:
        raise ValueError("Variation attack requires attack.parameters.num_averages > 0")
    if distance_metric not in SUPPORTED_DISTANCE_METRICS:
        raise ValueError(f"Unsupported variation distance metric: {distance_metric}")
    if threshold_strategy not in SUPPORTED_THRESHOLD_STRATEGIES:
        raise ValueError(f"Unsupported variation threshold strategy: {threshold_strategy}")

    return VariationPlan(
        entrypoint="variation-api",
        query_image_root=query_image_root,
        endpoint=endpoint,
        num_averages=num_averages,
        distance_metric=distance_metric,
        threshold_strategy=threshold_strategy,
    )


def _list_query_images(query_image_root: str | Path) -> list[Path]:
    query_root = Path(query_image_root)
    if not query_root.exists():
        return []
    return sorted(
        path
        for path in query_root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def probe_variation_assets(
    query_image_root: str | Path,
    endpoint: str,
) -> dict[str, object]:
    query_root = Path(query_image_root)
    query_images = _list_query_images(query_root)
    checks = {
        "query_image_root": query_root.exists(),
        "query_images_present": bool(query_images),
        "endpoint": bool(endpoint),
    }
    status = "ready" if all(checks.values()) else "blocked"
    paths = {
        "query_image_root": str(query_root),
        "endpoint": endpoint,
        "query_images": [str(path) for path in query_images],
    }
    labels = {
        "query_image_root": "query_image_root",
        "query_images_present": "query images",
        "endpoint": "endpoint",
    }
    missing_keys = [name for name, is_ready in checks.items() if not is_ready]
    missing = [
        str(query_root) if name == "query_image_root" else labels[name]
        for name in missing_keys
    ]
    return {
        "status": status,
        "checks": checks,
        "paths": paths,
        "missing": missing,
        "missing_keys": missing_keys,
        "missing_items": [Path(item).name if Path(item).name else item for item in missing],
        "missing_description": " / ".join(labels[name] for name in missing_keys),
    }


def explain_variation_assets(config: AuditConfig) -> dict[str, object]:
    plan = build_variation_plan(config)
    summary = probe_variation_assets(
        query_image_root=plan.query_image_root,
        endpoint=plan.endpoint,
    )
    return {
        **summary,
        "num_averages": plan.num_averages,
        "distance_metric": plan.distance_metric,
        "threshold_strategy": plan.threshold_strategy,
    }


def probe_variation_dry_run(config: AuditConfig) -> tuple[int, dict[str, object]]:
    try:
        plan = build_variation_plan(config)
        summary = probe_variation_assets(
            query_image_root=plan.query_image_root,
            endpoint=plan.endpoint,
        )
        checks = {
            **summary["checks"],
            "distance_metric_supported": plan.distance_metric in SUPPORTED_DISTANCE_METRICS,
            "threshold_strategy_supported": plan.threshold_strategy in SUPPORTED_THRESHOLD_STRATEGIES,
            "query_count_positive": len(summary["paths"]["query_images"]) > 0,
        }
    except ValueError as exc:
        return 1, {
            "status": "blocked",
            "error": str(exc),
        }

    status = "ready" if all(checks.values()) else "blocked"
    labels = {
        "query_image_root": "query_image_root",
        "query_images_present": "query images",
        "endpoint": "endpoint",
        "distance_metric_supported": "distance metric",
        "threshold_strategy_supported": "threshold strategy",
        "query_count_positive": "query count",
    }
    missing_keys = [name for name, is_ready in checks.items() if not is_ready]
    missing = list(summary["missing"])
    for key in missing_keys:
        if key not in summary["missing_keys"]:
            missing.append(labels[key])

    payload = {
        "status": status,
        "entrypoint": plan.entrypoint,
        "checks": checks,
        "paths": summary["paths"],
        "missing": missing,
        "missing_keys": missing_keys,
        "missing_items": [Path(item).name if Path(item).name else item for item in missing],
        "missing_description": " / ".join(labels[name] for name in missing_keys),
    }
    return (0 if status == "ready" else 1), payload


def run_variation_synthetic_smoke(workspace: str | Path) -> dict[str, object]:
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    synthetic_root = workspace_path / "synthetic-query-images"
    synthetic_root.mkdir(parents=True, exist_ok=True)
    for index in range(2):
        (synthetic_root / f"query_{index + 1}.png").write_bytes(b"png")

    member_distances = np.array([0.12, 0.18, 0.20, 0.15], dtype=float)
    nonmember_distances = np.array([0.81, 0.76, 0.84, 0.88], dtype=float)
    threshold = float(np.median(np.concatenate((member_distances, nonmember_distances))))
    asr = float(
        (
            np.concatenate(
                (
                    member_distances <= threshold,
                    nonmember_distances > threshold,
                )
            )
        ).mean()
    )
    wins = (member_distances[:, None] < nonmember_distances[None, :]).astype(float)
    ties = (member_distances[:, None] == nonmember_distances[None, :]).astype(float) * 0.5
    auc = float((wins + ties).mean())

    result = {
        "status": "ready",
        "track": "black-box",
        "method": "variation",
        "paper": "Towards_Black_Box_Diffusion_2024",
        "mode": "synthetic-smoke",
        "device": "cpu",
        "workspace": str(workspace_path),
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
        },
        "checks": {
            "synthetic_queries_created": True,
            "distance_scores_computed": True,
            "threshold_metrics_computed": True,
            "synthetic_assets_cleaned": True,
        },
        "metrics": {
            "auc": round(auc, 6),
            "asr": round(asr, 6),
            "threshold": round(threshold, 6),
        },
        "notes": [
            "Synthetic smoke emulates query-image variation distances without external API calls.",
            "Smoke validates the black-box threshold-style evaluation path only.",
        ],
    }
    (workspace_path / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    shutil.rmtree(synthetic_root, ignore_errors=True)
    return result
