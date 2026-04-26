"""Offline gray-box tri-score canary for X-88 CDI x TMIA-DM x PIA bridging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _round6(value: float) -> float:
    return round(float(value), 6)


def _orient_memberness(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    if float(member_scores.mean()) < float(nonmember_scores.mean()):
        return -member_scores, -nonmember_scores, "negated"
    return member_scores, nonmember_scores, "identity"


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    if std == 0.0:
        return np.zeros_like(values, dtype=float)
    return (values - float(values.mean())) / std


def _roc_curve_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="mergesort")
    scores_sorted = scores[order]
    labels_sorted = labels[order]
    positives = float(labels.sum())
    negatives = float(len(labels) - labels.sum())
    tps = np.cumsum(labels_sorted == 1)
    fps = np.cumsum(labels_sorted == 0)
    distinct = np.r_[True, scores_sorted[1:] != scores_sorted[:-1]]
    tpr = np.r_[0.0, tps[distinct] / positives, 1.0]
    fpr = np.r_[0.0, fps[distinct] / negatives, 1.0]
    return fpr, tpr


def _auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr = _roc_curve_points(scores, labels)
    return float(np.trapezoid(tpr, fpr))


def _accuracy_best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    sorted_scores = scores[order]
    thresholds = np.r_[
        sorted_scores[0] - 1e-6,
        (sorted_scores[:-1] + sorted_scores[1:]) / 2.0,
        sorted_scores[-1] + 1e-6,
    ]
    best = 0.0
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        best = max(best, float((preds == labels).mean()))
    return best


def _tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    fpr, tpr = _roc_curve_points(scores, labels)
    valid = tpr[fpr <= target_fpr]
    if valid.size == 0:
        return 0.0
    return float(valid.max())


def _metric_bundle(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "auc": _round6(_auc_score(scores, labels)),
        "asr": _round6(_accuracy_best_threshold(scores, labels)),
        "tpr_at_1pct_fpr": _round6(_tpr_at_fpr(scores, labels, 0.01)),
        "tpr_at_0_1pct_fpr": _round6(_tpr_at_fpr(scores, labels, 0.001)),
    }


def _load_pia_payload(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    return {
        "member_scores": np.asarray(payload["member_scores"], dtype=float),
        "nonmember_scores": np.asarray(payload["nonmember_scores"], dtype=float),
        "member_indices": [int(value) for value in payload["member_indices"]],
        "nonmember_indices": [int(value) for value in payload["nonmember_indices"]],
    }


def _load_tmiadm_family_payload(path: str | Path, family: str) -> dict[str, Any]:
    payload = _read_json(path)
    family_payload = payload[family]
    return {
        "member_scores": np.asarray(family_payload["member_scores"], dtype=float),
        "nonmember_scores": np.asarray(family_payload["nonmember_scores"], dtype=float),
        "metrics": family_payload.get("metrics", {}),
    }


def _extract_runtime_identity(summary_payload: dict[str, Any]) -> dict[str, Any]:
    runtime = summary_payload.get("runtime", {})
    return {
        "dataset_root": runtime.get("dataset_root"),
        "member_split_root": runtime.get("member_split_root"),
        "model_dir": runtime.get("model_dir"),
        "max_samples": runtime.get("max_samples", summary_payload.get("sample_count_per_split")),
        "num_samples": runtime.get("num_samples", summary_payload.get("sample_count_per_split")),
        "sample_count_per_split": summary_payload.get("sample_count_per_split"),
    }


def _normalize_identity_value(field: str, value: Any) -> Any:
    if value is None:
        return None
    if field in {"dataset_root", "member_split_root", "model_dir"}:
        return str(Path(str(value)).resolve(strict=False))
    return value


def load_identity_aligned_surface(
    *,
    name: str,
    pia_scores_path: str | Path,
    pia_summary_path: str | Path,
    tmiadm_family_scores_path: str | Path,
    tmiadm_summary_path: str | Path,
    tmiadm_family: str,
) -> dict[str, Any]:
    pia_payload = _load_pia_payload(pia_scores_path)
    tmiadm_payload = _load_tmiadm_family_payload(tmiadm_family_scores_path, tmiadm_family)

    if len(pia_payload["member_scores"]) != len(tmiadm_payload["member_scores"]):
        raise ValueError(
            f"Identity alignment mismatch for {name}: member count differs between PIA and TMIA-DM."
        )
    if len(pia_payload["nonmember_scores"]) != len(tmiadm_payload["nonmember_scores"]):
        raise ValueError(
            f"Identity alignment mismatch for {name}: nonmember count differs between PIA and TMIA-DM."
        )

    pia_summary = _read_json(pia_summary_path)
    tmiadm_summary = _read_json(tmiadm_summary_path)
    pia_identity = _extract_runtime_identity(pia_summary)
    tmiadm_identity = _extract_runtime_identity(tmiadm_summary)

    matching_fields: dict[str, Any] = {}
    for field in ("dataset_root", "member_split_root", "model_dir", "max_samples", "num_samples", "sample_count_per_split"):
        pia_value = pia_identity.get(field)
        tmiadm_value = tmiadm_identity.get(field)
        normalized_pia = _normalize_identity_value(field, pia_value)
        normalized_tmiadm = _normalize_identity_value(field, tmiadm_value)
        if normalized_pia != normalized_tmiadm:
            raise ValueError(
                f"Identity alignment mismatch for {name}: field '{field}' differs "
                f"(PIA={pia_value!r}, TMIA-DM={tmiadm_value!r})."
            )
        matching_fields[field] = normalized_pia

    return {
        "name": name,
        "pia_scores_path": str(Path(pia_scores_path)),
        "pia_summary_path": str(Path(pia_summary_path)),
        "tmiadm_family_scores_path": str(Path(tmiadm_family_scores_path)),
        "tmiadm_summary_path": str(Path(tmiadm_summary_path)),
        "tmiadm_family": tmiadm_family,
        "member_indices": pia_payload["member_indices"],
        "nonmember_indices": pia_payload["nonmember_indices"],
        "pia_member_scores": pia_payload["member_scores"],
        "pia_nonmember_scores": pia_payload["nonmember_scores"],
        "tmiadm_member_scores": tmiadm_payload["member_scores"],
        "tmiadm_nonmember_scores": tmiadm_payload["nonmember_scores"],
        "analysis": {
            "identity_alignment": {
                "status": "frozen",
                "tmiadm_index_strategy": "adopt-pia-order",
                "matching_runtime_fields": matching_fields,
            }
        },
    }


def _build_control_z_linear(
    feature_scores: dict[str, np.ndarray],
    member_count: int,
) -> dict[str, Any]:
    control_size = member_count // 2
    control_gaps: dict[str, float] = {}
    for feature_name, scores in feature_scores.items():
        member_scores = scores[:member_count]
        nonmember_scores = scores[member_count:]
        control_gap = float(member_scores[:control_size].mean() - nonmember_scores[:control_size].mean())
        control_gaps[feature_name] = max(control_gap, 0.0)
    total_gap = float(sum(control_gaps.values()))
    if total_gap <= 1e-12:
        weight = 1.0 / float(len(feature_scores))
        weights = {name: weight for name in feature_scores}
    else:
        weights = {name: control_gaps[name] / total_gap for name in feature_scores}
    combined = sum(weights[name] * feature_scores[name] for name in feature_scores)
    return {
        "score": combined,
        "weights": weights,
        "control_gaps": control_gaps,
    }


def evaluate_identity_aligned_surface(surface: dict[str, Any]) -> dict[str, Any]:
    pia_member, pia_nonmember, pia_orientation = _orient_memberness(
        surface["pia_member_scores"],
        surface["pia_nonmember_scores"],
    )
    tmiadm_member, tmiadm_nonmember, tmiadm_orientation = _orient_memberness(
        surface["tmiadm_member_scores"],
        surface["tmiadm_nonmember_scores"],
    )

    labels = np.r_[
        np.ones(len(pia_member), dtype=int),
        np.zeros(len(pia_nonmember), dtype=int),
    ]
    pia_scores = np.r_[pia_member, pia_nonmember]
    tmiadm_scores = np.r_[tmiadm_member, tmiadm_nonmember]
    pia_z = _zscore(pia_scores)
    tmiadm_z = _zscore(tmiadm_scores)
    zscore_sum = pia_z + tmiadm_z

    feature_scores = {
        "pia": pia_z,
        "tmiadm": tmiadm_z,
        "zscore_sum": zscore_sum,
    }
    control_z_linear = _build_control_z_linear(feature_scores, member_count=len(pia_member))
    consensus_floor = _zscore(np.minimum(pia_z, tmiadm_z))
    consensus_penalty = min(control_z_linear["weights"].values())
    composite = control_z_linear["score"] - (consensus_penalty * consensus_floor)

    component_metrics = {name: _metric_bundle(score, labels) for name, score in feature_scores.items()}
    baseline_metrics = {
        "zscore_sum": component_metrics["zscore_sum"],
        "control_z_linear": _metric_bundle(control_z_linear["score"], labels),
    }
    composite_metrics = _metric_bundle(composite, labels)

    return {
        **surface,
        "metrics": {
            "composite": composite_metrics,
            "component_metrics": component_metrics,
            "component_auc": {
                "pia": component_metrics["pia"]["auc"],
                "tmiadm": component_metrics["tmiadm"]["auc"],
                "zscore_sum": component_metrics["zscore_sum"]["auc"],
            },
            "baselines": baseline_metrics,
        },
        "analysis": {
            **surface["analysis"],
            "feature_orientations": {
                "pia": pia_orientation,
                "tmiadm": tmiadm_orientation,
            },
            "tri_scorer_details": {
                "base_scorer": "control-z-linear",
                "base_weights": {name: _round6(value) for name, value in control_z_linear["weights"].items()},
                "control_gaps": {name: _round6(value) for name, value in control_z_linear["control_gaps"].items()},
                "consensus_floor_correction": {
                    "mode": "subtract-min-weight-times-zscore-min-component",
                    "penalty_weight": _round6(consensus_penalty),
                },
            },
        },
    }


def _mean_metric(surfaces: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values = []
    for surface in surfaces:
        cursor: Any = surface
        for key in path:
            cursor = cursor[key]
        values.append(float(cursor))
    return _round6(sum(values) / float(len(values)))


def _decide_verdict(
    evaluated_surfaces: list[dict[str, Any]],
    auc_drop_tolerance: float = 0.005,
) -> tuple[str, dict[str, Any]]:
    winning_surfaces: list[dict[str, Any]] = []
    for surface in evaluated_surfaces:
        composite = surface["metrics"]["composite"]
        zsum = surface["metrics"]["baselines"]["zscore_sum"]
        control = surface["metrics"]["baselines"]["control_z_linear"]
        low_fpr_targets = {
            "tpr_at_1pct_fpr": composite["tpr_at_1pct_fpr"] > zsum["tpr_at_1pct_fpr"]
            and composite["tpr_at_1pct_fpr"] > control["tpr_at_1pct_fpr"],
            "tpr_at_0_1pct_fpr": composite["tpr_at_0_1pct_fpr"] > zsum["tpr_at_0_1pct_fpr"]
            and composite["tpr_at_0_1pct_fpr"] > control["tpr_at_0_1pct_fpr"],
        }
        auc_ok = composite["auc"] >= max(zsum["auc"], control["auc"]) - float(auc_drop_tolerance)
        if auc_ok and any(low_fpr_targets.values()):
            winning_surfaces.append(
                {
                    "name": surface["name"],
                    "low_fpr_targets": low_fpr_targets,
                    "auc_ok": auc_ok,
                }
            )

    if winning_surfaces:
        return (
            "positive",
            {
                "kill_gate_passed": True,
                "auc_drop_tolerance": float(auc_drop_tolerance),
                "winning_surfaces": winning_surfaces,
            },
        )
    return (
        "no-go",
        {
            "kill_gate_passed": False,
            "auc_drop_tolerance": float(auc_drop_tolerance),
            "winning_surfaces": [],
        },
    )


def _build_contract(contract_reference_path: str | Path) -> dict[str, Any]:
    reference = _read_json(contract_reference_path)
    reference_contract = reference.get("contract", {})
    return {
        "name": reference_contract.get("name", "cdi-paired-canary-summary"),
        "version": "2026-04-17-x88-triscore-v1",
        "feature_mode": "paired-pia-tmiadm-zscore-triscore",
        "component_reporting_required": True,
        "headline_use_allowed": bool(reference_contract.get("headline_use_allowed", False)),
        "external_evidence_allowed": bool(reference_contract.get("external_evidence_allowed", False)),
    }


def _aggregate_analysis(evaluated_surfaces: list[dict[str, Any]], verdict_details: dict[str, Any]) -> dict[str, Any]:
    return {
        "identity_alignment": {
            "status": "frozen",
            "surfaces": {
                surface["name"]: surface["analysis"]["identity_alignment"] for surface in evaluated_surfaces
            },
        },
        "feature_orientations": {
            surface["name"]: surface["analysis"]["feature_orientations"] for surface in evaluated_surfaces
        },
        "tri_scorer_details": {
            "scorer_name": "control-z-linear-minus-consensus-floor",
            "formula": "control_z_linear - min(base_weights) * zscore(min(pia_z, tmiadm_z))",
            "per_surface": {
                surface["name"]: surface["analysis"]["tri_scorer_details"] for surface in evaluated_surfaces
            },
            "kill_gate": verdict_details,
        },
    }


def _serializable_surface(surface: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": surface["name"],
        "pia_scores_path": surface["pia_scores_path"],
        "pia_summary_path": surface["pia_summary_path"],
        "tmiadm_family_scores_path": surface["tmiadm_family_scores_path"],
        "tmiadm_summary_path": surface["tmiadm_summary_path"],
        "tmiadm_family": surface["tmiadm_family"],
        "member_indices": surface["member_indices"],
        "nonmember_indices": surface["nonmember_indices"],
        "metrics": surface["metrics"],
        "analysis": surface["analysis"],
    }


def run_graybox_triscore_canary(
    *,
    run_root: str | Path,
    contract_reference_path: str | Path,
    surfaces: list[dict[str, Any]],
) -> dict[str, Any]:
    run_root_path = Path(run_root)
    run_root_path.mkdir(parents=True, exist_ok=True)
    contract = _build_contract(contract_reference_path)

    evaluated_surfaces = [
        evaluate_identity_aligned_surface(load_identity_aligned_surface(**surface_spec))
        for surface_spec in surfaces
    ]
    verdict, verdict_details = _decide_verdict(evaluated_surfaces)

    summary = {
        "status": "completed",
        "track": "gray-box",
        "method": "cdi-tmiadm-triscore-canary",
        "mode": "offline-canary",
        "run_id": run_root_path.name,
        "contract": contract,
        "metrics": {
            "auc": _mean_metric(evaluated_surfaces, ("metrics", "composite", "auc")),
            "asr": _mean_metric(evaluated_surfaces, ("metrics", "composite", "asr")),
            "tpr_at_1pct_fpr": _mean_metric(evaluated_surfaces, ("metrics", "composite", "tpr_at_1pct_fpr")),
            "tpr_at_0_1pct_fpr": _mean_metric(evaluated_surfaces, ("metrics", "composite", "tpr_at_0_1pct_fpr")),
            "component_auc": {
                "pia": _mean_metric(evaluated_surfaces, ("metrics", "component_auc", "pia")),
                "tmiadm": _mean_metric(evaluated_surfaces, ("metrics", "component_auc", "tmiadm")),
                "zscore_sum": _mean_metric(evaluated_surfaces, ("metrics", "component_auc", "zscore_sum")),
            },
        },
        "analysis": _aggregate_analysis(evaluated_surfaces, verdict_details),
        "surfaces": [_serializable_surface(surface) for surface in evaluated_surfaces],
        "verdict": verdict,
        "notes": [
            "This canary stays internal-only and preserves the CDI contract-first boundary.",
            "TMIA-DM sample identity is frozen by adopting the aligned PIA packet order only after runtime-contract equality checks pass.",
            "The tri-score is a new offline scorer; it is not a threshold-scan restatement of the earlier switching packet.",
        ],
        "artifacts": {
            "audit_summary": str((run_root_path / "audit_summary.json").as_posix()),
        },
    }
    (run_root_path / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary
