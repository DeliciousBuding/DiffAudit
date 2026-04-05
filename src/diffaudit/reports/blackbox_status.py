"""Aggregate black-box experiment summaries into a single status report."""

from __future__ import annotations

import json
from pathlib import Path


BLACKBOX_EVIDENCE_RANK = {
    "dry-run-smoke": 1,
    "synthetic-smoke": 2,
    "eval-smoke": 3,
    "artifact-summary": 4,
    "upstream-eval-smoke": 5,
}


def _extract_headline_metrics(payload: dict[str, object]) -> dict[str, float | None]:
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        return {"auc": None, "asr": None, "tpr_at_1pct_fpr": None}

    auc = None
    asr = None
    tpr = None
    if "auc" in metrics:
        auc = metrics.get("auc")
    if "asr" in metrics:
        asr = metrics.get("asr")
    if "tpr_at_1pct_fpr" in metrics:
        tpr = metrics.get("tpr_at_1pct_fpr")

    target_metrics = metrics.get("target")
    if isinstance(target_metrics, dict):
        auc = target_metrics.get("auc", auc)
        asr = target_metrics.get("best_asr", asr)
        tpr = target_metrics.get("tpr_at_1pct_fpr", tpr)

    if "reported_auc" in metrics:
        auc = metrics.get("reported_auc", auc)
    if "reported_accuracy" in metrics and asr is None:
        asr = metrics.get("reported_accuracy")

    return {
        "auc": auc,
        "asr": asr,
        "tpr_at_1pct_fpr": tpr,
    }


def _evidence_rank(mode: str) -> int:
    return BLACKBOX_EVIDENCE_RANK.get(mode, 0)


def build_blackbox_status_report(
    experiments_root: str | Path,
    workspace: str | Path,
) -> dict[str, object]:
    experiments_path = Path(experiments_root)
    summary_paths = sorted(experiments_path.glob("*/summary.json"))
    blackbox_payloads: list[dict[str, object]] = []
    for summary_path in summary_paths:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("track") == "black-box":
            blackbox_payloads.append(payload | {"_summary_path": str(summary_path)})

    methods: dict[str, dict[str, object]] = {}
    for payload in blackbox_payloads:
        method = str(payload["method"])
        entry = methods.setdefault(
            method,
            {
                "paper": payload.get("paper"),
                "best_evidence_mode": payload.get("mode"),
                "best_evidence_path": payload["_summary_path"],
                "headline_metrics": _extract_headline_metrics(payload),
                "evidence": [],
            },
        )
        entry["evidence"].append(
            {
                "mode": payload.get("mode"),
                "status": payload.get("status"),
                "summary_path": payload["_summary_path"],
                "headline_metrics": _extract_headline_metrics(payload),
            }
        )
        current_rank = _evidence_rank(str(payload.get("mode", "")))
        best_rank = _evidence_rank(str(entry["best_evidence_mode"]))
        if current_rank >= best_rank:
            entry["best_evidence_mode"] = payload.get("mode")
            entry["best_evidence_path"] = payload["_summary_path"]
            entry["headline_metrics"] = _extract_headline_metrics(payload)

    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "ready",
        "track": "black-box",
        "method_count": len(methods),
        "methods": methods,
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
        },
        "notes": [
            "This report aggregates all black-box experiment summaries found under the experiments root.",
            "best_evidence_mode is ranked by execution depth, not by metric magnitude.",
        ],
    }
    (workspace_path / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return report
