"""Minimal smoke pipeline for early audit experiments."""

from __future__ import annotations

import json
from pathlib import Path

from diffaudit.config import AuditConfig


def build_smoke_summary(config: AuditConfig) -> dict[str, object]:
    return {
        "run_name": config.task.name,
        "model_family": config.task.model_family,
        "access_level": config.task.access_level,
        "dataset_id": config.assets.dataset_id,
        "model_id": config.assets.model_id,
        "attack_method": config.attack.method,
        "num_samples": config.attack.num_samples,
        "output_dir": config.report.output_dir,
    }


def run_smoke_pipeline(config: AuditConfig, workspace: str | Path) -> Path:
    summary = build_smoke_summary(config)
    workspace_path = Path(workspace)
    output_dir = workspace_path / config.report.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary_path
