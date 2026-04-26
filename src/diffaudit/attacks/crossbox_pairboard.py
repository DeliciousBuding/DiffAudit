"""Generic shared-score pairboard evaluator for cross-box analysis."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

PAIRBOARD_METRICS = (
    "auc",
    "asr",
    "tpr_at_1pct_fpr",
    "tpr_at_0_1pct_fpr",
)
REPEATED_HOLDOUT_SEED_STRIDE = 2
DEFAULT_CASCADE_ROUTE_FRACTIONS = (0.1, 0.2, 0.3)
DEFAULT_CASCADE_GAMMAS = (0.0, 0.1, 0.2, 0.3)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _round6(value: float) -> float:
    return round(float(value), 6)


def _rounded_array(values: list[float] | np.ndarray) -> np.ndarray:
    return np.asarray([_round6(value) for value in np.asarray(values, dtype=float).tolist()], dtype=float)


def _load_tensor_scores(path: str | Path) -> np.ndarray:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor score artifact: {path}")
    return _rounded_array(payload.detach().cpu().reshape(-1).tolist())


def _load_record_ids(path: str | Path) -> list[int] | None:
    records_path = Path(path)
    if not records_path.exists():
        return None
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        return None

    sample_path_ids: list[int] = []
    for record in records:
        sample_path = record.get("sample_path")
        if isinstance(sample_path, str):
            match = re.search(r"-(\d+)\.[^.]+$", sample_path)
            if match is not None:
                sample_path_ids.append(int(match.group(1)))
                continue
        break
    if len(sample_path_ids) == len(records):
        return sample_path_ids

    sample_index_ids: list[int] = []
    for record in records:
        sample_index = record.get("sample_index")
        if sample_index is None:
            return None
        sample_index_ids.append(int(sample_index))
    return sample_index_ids


def _validate_surface_payload(payload: dict[str, Any], source_path: Path) -> None:
    if "member_scores" not in payload or "nonmember_scores" not in payload:
        raise ValueError(f"Pairboard surface is missing member/nonmember scores: {source_path}")


def load_pairboard_surface(
    source_path: str | Path,
    *,
    name: str | None = None,
    family: str | None = None,
) -> dict[str, Any]:
    resolved = Path(source_path)
    payload = _read_json(resolved)
    surface_name = name or resolved.stem

    if payload.get("mode") == "loss-score-export":
        if payload.get("status") != "ready":
            raise ValueError(f"GSA loss-score export summary is not ready: {resolved}")
        exports = payload.get("exports", {})
        target_member_export = exports.get("target_member", {})
        target_nonmember_export = exports.get("target_non_member", {})
        target_member = target_member_export.get("output_path")
        target_nonmember = target_nonmember_export.get("output_path")
        if not target_member or not target_nonmember:
            raise ValueError(f"GSA loss-score export summary is missing target score paths: {resolved}")
        return {
            "name": surface_name,
            "source_path": str(resolved),
            "source_kind": "gsa-loss-score-export",
            "family": None,
            "member_scores": _load_tensor_scores(target_member),
            "nonmember_scores": _load_tensor_scores(target_nonmember),
            "member_indices": _load_record_ids(target_member_export.get("records_path"))
            if target_member_export.get("records_path")
            else None,
            "nonmember_indices": _load_record_ids(target_nonmember_export.get("records_path"))
            if target_nonmember_export.get("records_path")
            else None,
        }

    if family is not None:
        if family not in payload:
            raise ValueError(f"Requested family '{family}' was not found in {resolved}")
        family_payload = payload[family]
        if not isinstance(family_payload, dict):
            raise ValueError(f"Requested family '{family}' in {resolved} must be a JSON object")
        _validate_surface_payload(family_payload, resolved)
        payload = family_payload
        source_kind = "family-json"
    else:
        _validate_surface_payload(payload, resolved)
        source_kind = "generic-json"

    member_indices = payload.get("member_indices")
    nonmember_indices = payload.get("nonmember_indices")
    return {
        "name": surface_name,
        "source_path": str(resolved),
        "source_kind": source_kind,
        "family": family,
        "member_scores": _rounded_array(payload["member_scores"]),
        "nonmember_scores": _rounded_array(payload["nonmember_scores"]),
        "member_indices": [int(value) for value in member_indices] if member_indices is not None else None,
        "nonmember_indices": [int(value) for value in nonmember_indices] if nonmember_indices is not None else None,
    }


def _align_split(
    primary_scores: np.ndarray,
    primary_indices: list[int] | None,
    secondary_scores: np.ndarray,
    secondary_indices: list[int] | None,
) -> tuple[np.ndarray, np.ndarray, list[int] | None, str]:
    if primary_indices is not None and secondary_indices is not None:
        secondary_lookup = {int(index): position for position, index in enumerate(secondary_indices)}
        shared_indices = [int(index) for index in primary_indices if int(index) in secondary_lookup]
        if not shared_indices:
            raise ValueError("Pairboard surfaces do not share any indices on the requested split")
        primary_positions = [position for position, index in enumerate(primary_indices) if int(index) in secondary_lookup]
        secondary_positions = [secondary_lookup[index] for index in shared_indices]
        return (
            primary_scores[np.asarray(primary_positions, dtype=int)],
            secondary_scores[np.asarray(secondary_positions, dtype=int)],
            shared_indices,
            "shared-index-intersection",
        )

    if primary_scores.shape[0] != secondary_scores.shape[0]:
        raise ValueError("Pairboard surfaces without shared indices must have equal per-split lengths")

    retained_indices = primary_indices if primary_indices is not None else secondary_indices
    strategy = "positional-order" if retained_indices is None else "positional-order-single-index"
    return primary_scores.copy(), secondary_scores.copy(), retained_indices, strategy


def _align_surfaces(surface_a: dict[str, Any], surface_b: dict[str, Any]) -> dict[str, Any]:
    member_a, member_b, member_indices, member_strategy = _align_split(
        surface_a["member_scores"],
        surface_a["member_indices"],
        surface_b["member_scores"],
        surface_b["member_indices"],
    )
    nonmember_a, nonmember_b, nonmember_indices, nonmember_strategy = _align_split(
        surface_a["nonmember_scores"],
        surface_a["nonmember_indices"],
        surface_b["nonmember_scores"],
        surface_b["nonmember_indices"],
    )
    return {
        "member_a": member_a,
        "member_b": member_b,
        "nonmember_a": nonmember_a,
        "nonmember_b": nonmember_b,
        "member_indices": member_indices,
        "nonmember_indices": nonmember_indices,
        "alignment": {
            "member_strategy": member_strategy,
            "nonmember_strategy": nonmember_strategy,
            "shared_member_count": int(member_a.shape[0]),
            "shared_nonmember_count": int(nonmember_a.shape[0]),
        },
    }


def _split_positions(count: int, calibration_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if count < 2:
        raise ValueError("Pairboard evaluation requires at least two samples per split")
    requested = int(round(float(count) * float(calibration_fraction)))
    calibration_count = min(max(requested, 1), count - 1)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(count)
    calibration = np.sort(order[:calibration_count])
    test = np.sort(order[calibration_count:])
    return calibration, test


def _orient_memberness(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    if float(member_scores.mean()) < float(nonmember_scores.mean()):
        return -member_scores, -nonmember_scores, "negated"
    return member_scores.copy(), nonmember_scores.copy(), "identity"


def _fit_standardizer(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> dict[str, float]:
    pooled = np.concatenate([member_scores, nonmember_scores])
    mean = float(pooled.mean())
    std = float(pooled.std())
    return {
        "mean": mean,
        "std": std if std > 0.0 else 1.0,
    }


def _apply_standardizer(values: np.ndarray, standardizer: dict[str, float]) -> np.ndarray:
    return (values - float(standardizer["mean"])) / float(standardizer["std"])


def _roc_curve_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="mergesort")
    scores_sorted = scores[order]
    labels_sorted = labels[order]
    positives = float(labels.sum())
    negatives = float(labels.shape[0] - labels.sum())
    tps = np.cumsum(labels_sorted == 1)
    fps = np.cumsum(labels_sorted == 0)
    distinct = np.r_[True, scores_sorted[1:] != scores_sorted[:-1]]
    tpr = np.r_[0.0, tps[distinct] / positives, 1.0]
    fpr = np.r_[0.0, fps[distinct] / negatives, 1.0]
    return fpr, tpr


def _auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr = _roc_curve_points(scores, labels)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) * 0.5))


def _accuracy_best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    sorted_scores = np.sort(scores)
    thresholds = np.r_[
        sorted_scores[0] - 1e-6,
        (sorted_scores[:-1] + sorted_scores[1:]) / 2.0,
        sorted_scores[-1] + 1e-6,
    ]
    best = 0.0
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        best = max(best, float((predictions == labels).mean()))
    return best


def _tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    fpr, tpr = _roc_curve_points(scores, labels)
    valid = tpr[fpr <= float(target_fpr)]
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


def _normalize_float_grid(
    values: list[float] | tuple[float, ...] | str | None,
    *,
    default: tuple[float, ...],
    lower_bound: float,
    upper_bound: float,
    allow_zero: bool = False,
) -> list[float]:
    if values is None:
        resolved = [float(value) for value in default]
    elif isinstance(values, str):
        resolved = [float(token.strip()) for token in values.split(",") if token.strip()]
    else:
        resolved = [float(value) for value in values]
    if not resolved:
        raise ValueError("Float grid must contain at least one value")
    normalized: list[float] = []
    for value in resolved:
        if allow_zero:
            valid = lower_bound <= value <= upper_bound
        else:
            valid = lower_bound < value <= upper_bound
        if not valid:
            raise ValueError(f"Grid value out of range: {value}")
        if value not in normalized:
            normalized.append(float(value))
    return normalized


def _fit_probability_calibrator(scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(np.asarray(scores, dtype=float), np.asarray(labels, dtype=float))
    return calibrator


def _threshold_at_fpr_constraint(
    scores: np.ndarray,
    labels: np.ndarray,
    target_fpr: float,
) -> dict[str, float]:
    sorted_scores = np.sort(np.asarray(scores, dtype=float))
    thresholds = np.r_[
        sorted_scores[0] - 1e-6,
        (sorted_scores[:-1] + sorted_scores[1:]) / 2.0,
        sorted_scores[-1] + 1e-6,
    ]
    positives = float(labels.sum())
    negatives = float(labels.shape[0] - labels.sum())
    best_threshold = float(thresholds[-1])
    best_tpr = -1.0
    best_fpr = 0.0
    for threshold in thresholds:
        predictions = scores >= threshold
        tp = float(((predictions == 1) & (labels == 1)).sum())
        fp = float(((predictions == 1) & (labels == 0)).sum())
        current_tpr = tp / positives if positives > 0.0 else 0.0
        current_fpr = fp / negatives if negatives > 0.0 else 0.0
        if current_fpr <= float(target_fpr) + 1e-12:
            if current_tpr > best_tpr + 1e-12 or (
                abs(current_tpr - best_tpr) <= 1e-12 and threshold < best_threshold
            ):
                best_threshold = float(threshold)
                best_tpr = float(current_tpr)
                best_fpr = float(current_fpr)
    return {
        "threshold": _round6(best_threshold),
        "tpr": _round6(max(best_tpr, 0.0)),
        "fpr": _round6(best_fpr),
    }


def _delta_for_route_fraction(margins: np.ndarray, target_route_fraction: float) -> float:
    requested = max(1, int(round(float(margins.shape[0]) * float(target_route_fraction))))
    sorted_margins = np.sort(np.asarray(margins, dtype=float))
    return float(sorted_margins[min(requested - 1, sorted_margins.shape[0] - 1)])


def _build_tail_gated_cascade_candidate(
    *,
    calibration_scores: dict[str, np.ndarray],
    test_scores: dict[str, np.ndarray],
    calibration_labels: np.ndarray,
    test_labels: np.ndarray,
    anchor_name: str,
    candidate_name: str,
    surface_a_name: str,
    surface_b_name: str,
    route_fractions: list[float] | tuple[float, ...] | str | None,
    gammas: list[float] | tuple[float, ...] | str | None,
    secondary_cost_ratio: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    if anchor_name not in calibration_scores:
        raise ValueError(f"Tail-gated cascade anchor '{anchor_name}' is unavailable")
    if candidate_name not in calibration_scores:
        raise ValueError(f"Tail-gated cascade candidate '{candidate_name}' is unavailable")
    if surface_a_name not in calibration_scores or surface_b_name not in calibration_scores:
        raise ValueError("Tail-gated cascade requires both raw surface scores for disagreement tracking")
    if float(secondary_cost_ratio) < 0.0:
        raise ValueError("Tail-gated cascade secondary_cost_ratio must be non-negative")

    resolved_route_fractions = _normalize_float_grid(
        route_fractions,
        default=DEFAULT_CASCADE_ROUTE_FRACTIONS,
        lower_bound=0.0,
        upper_bound=1.0,
    )
    resolved_gammas = _normalize_float_grid(
        gammas,
        default=DEFAULT_CASCADE_GAMMAS,
        lower_bound=0.0,
        upper_bound=1.0,
        allow_zero=True,
    )

    calibration_probabilities: dict[str, np.ndarray] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    for name, scores in calibration_scores.items():
        calibrator = _fit_probability_calibrator(scores, calibration_labels)
        calibration_probabilities[name] = np.asarray(calibrator.predict(scores), dtype=float)
        test_probabilities[name] = np.asarray(calibrator.predict(test_scores[name]), dtype=float)

    anchor_calibration_prob = calibration_probabilities[anchor_name]
    anchor_test_prob = test_probabilities[anchor_name]
    candidate_calibration_prob = calibration_probabilities[candidate_name]
    candidate_test_prob = test_probabilities[candidate_name]
    disagreement_calibration = np.abs(
        calibration_probabilities[surface_a_name] - calibration_probabilities[surface_b_name]
    )
    disagreement_test = np.abs(
        test_probabilities[surface_a_name] - test_probabilities[surface_b_name]
    )

    anchor_tau_1pct = _threshold_at_fpr_constraint(anchor_calibration_prob, calibration_labels, 0.01)
    anchor_tau_0_1pct = _threshold_at_fpr_constraint(anchor_calibration_prob, calibration_labels, 0.001)
    margins = np.abs(anchor_calibration_prob - float(anchor_tau_1pct["threshold"]))

    best_selection: dict[str, Any] | None = None
    best_calibration_scores: np.ndarray | None = None
    selection_rows: list[dict[str, Any]] = []
    for target_route_fraction in resolved_route_fractions:
        delta = _delta_for_route_fraction(margins, target_route_fraction)
        calibration_route_mask = margins <= delta
        for gamma in resolved_gammas:
            routed_calibration_scores = np.clip(
                candidate_calibration_prob - float(gamma) * disagreement_calibration,
                0.0,
                1.0,
            )
            cascade_calibration_scores = anchor_calibration_prob.copy()
            cascade_calibration_scores[calibration_route_mask] = routed_calibration_scores[calibration_route_mask]
            metrics = _metric_bundle(cascade_calibration_scores, calibration_labels)
            actual_routed_fraction = float(calibration_route_mask.mean())
            selection_rows.append(
                {
                    "target_route_fraction": _round6(target_route_fraction),
                    "actual_calibration_route_fraction": _round6(actual_routed_fraction),
                    "gamma": _round6(gamma),
                    "band_delta": _round6(delta),
                    "relative_overhead": _round6(actual_routed_fraction * float(secondary_cost_ratio)),
                    "metrics": metrics,
                }
            )
            ranking_key = (
                metrics["tpr_at_1pct_fpr"],
                metrics["tpr_at_0_1pct_fpr"],
                metrics["auc"],
                metrics["asr"],
                -(actual_routed_fraction * float(secondary_cost_ratio)),
            )
            if best_selection is None or ranking_key > best_selection["ranking_key"]:
                best_selection = {
                    "ranking_key": ranking_key,
                    "target_route_fraction": float(target_route_fraction),
                    "actual_calibration_route_fraction": float(actual_routed_fraction),
                    "gamma": float(gamma),
                    "band_delta": float(delta),
                    "relative_overhead": float(actual_routed_fraction * float(secondary_cost_ratio)),
                    "calibration_metrics": metrics,
                    "calibration_routed_count": int(calibration_route_mask.sum()),
                }
                best_calibration_scores = cascade_calibration_scores

    if best_selection is None or best_calibration_scores is None:
        raise ValueError("Tail-gated cascade selection failed to produce any candidate")

    test_route_mask = np.abs(anchor_test_prob - float(anchor_tau_1pct["threshold"])) <= float(best_selection["band_delta"])
    routed_test_scores = np.clip(
        candidate_test_prob - float(best_selection["gamma"]) * disagreement_test,
        0.0,
        1.0,
    )
    cascade_test_scores = anchor_test_prob.copy()
    cascade_test_scores[test_route_mask] = routed_test_scores[test_route_mask]
    test_metrics = _metric_bundle(cascade_test_scores, test_labels)
    actual_test_route_fraction = float(test_route_mask.mean())

    analysis = {
        "selection": {
            "anchor_name": anchor_name,
            "candidate_name": candidate_name,
            "target_route_fraction": _round6(best_selection["target_route_fraction"]),
            "actual_calibration_route_fraction": _round6(best_selection["actual_calibration_route_fraction"]),
            "gamma": _round6(best_selection["gamma"]),
            "secondary_cost_ratio": _round6(float(secondary_cost_ratio)),
            "band_delta": _round6(best_selection["band_delta"]),
            "anchor_tau_1pct": anchor_tau_1pct,
            "anchor_tau_0_1pct": anchor_tau_0_1pct,
            "selection_metric": "tpr_at_1pct_fpr",
        },
        "calibration": {
            "routed_count": int(best_selection["calibration_routed_count"]),
            "routed_fraction": _round6(best_selection["actual_calibration_route_fraction"]),
            "relative_overhead": _round6(best_selection["relative_overhead"]),
            "metrics": best_selection["calibration_metrics"],
        },
        "test": {
            "routed_count": int(test_route_mask.sum()),
            "routed_fraction": _round6(actual_test_route_fraction),
            "relative_overhead": _round6(actual_test_route_fraction * float(secondary_cost_ratio)),
            "metrics": test_metrics,
        },
        "scan": selection_rows,
    }
    return best_selection["calibration_metrics"], test_metrics, analysis


def _choose_best_single(
    candidate_scores: dict[str, np.ndarray],
    labels: np.ndarray,
) -> tuple[str, dict[str, float]]:
    ranked = sorted(
        (
            (name, _metric_bundle(scores, labels))
            for name, scores in candidate_scores.items()
        ),
        key=lambda item: (
            item[1]["auc"],
            item[1]["tpr_at_0_1pct_fpr"],
            item[1]["tpr_at_1pct_fpr"],
            item[1]["asr"],
        ),
        reverse=True,
    )
    return ranked[0]


def _weighted_average(
    calibration_features: dict[str, np.ndarray],
    calibration_labels: np.ndarray,
    evaluation_features: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    gaps = {}
    for name, scores in calibration_features.items():
        member_scores = scores[calibration_labels == 1]
        nonmember_scores = scores[calibration_labels == 0]
        gaps[name] = max(float(member_scores.mean() - nonmember_scores.mean()), 0.0)
    total_gap = sum(gaps.values())
    if total_gap <= 1e-12:
        weight = 1.0 / float(len(calibration_features))
        weights = {name: weight for name in calibration_features}
    else:
        weights = {name: gaps[name] / total_gap for name in calibration_features}
    combined = np.zeros_like(next(iter(evaluation_features.values())))
    for name, scores in evaluation_features.items():
        combined = combined + float(weights[name]) * scores
    return combined, {
        "weights": {name: _round6(value) for name, value in weights.items()},
        "calibration_gaps": {name: _round6(value) for name, value in gaps.items()},
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_logistic_2feature(
    features: np.ndarray,
    labels: np.ndarray,
    steps: int = 500,
    learning_rate: float = 0.2,
    l2_weight: float = 1e-3,
) -> tuple[np.ndarray, float]:
    weights = np.zeros(features.shape[1], dtype=float)
    bias = 0.0
    for _ in range(int(steps)):
        logits = features @ weights + bias
        probabilities = _sigmoid(logits)
        error = probabilities - labels
        gradient_weights = (features.T @ error) / float(features.shape[0]) + float(l2_weight) * weights
        gradient_bias = float(error.mean())
        weights = weights - float(learning_rate) * gradient_weights
        bias = bias - float(learning_rate) * gradient_bias
    return weights, bias


def _support_disconfirm_neutral(
    feature_a: np.ndarray,
    feature_b: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    support = np.minimum(feature_a, feature_b)
    disconfirm = np.maximum(-feature_a, -feature_b)
    consensus = -np.abs(feature_a - feature_b)
    score = support - disconfirm + 0.25 * consensus
    support_mask = (feature_a >= 0.0) & (feature_b >= 0.0)
    disconfirm_mask = (feature_a < 0.0) & (feature_b < 0.0)
    neutral_mask = ~(support_mask | disconfirm_mask)
    return score, {
        "support": int(support_mask.sum()),
        "disconfirm": int(disconfirm_mask.sum()),
        "neutral": int(neutral_mask.sum()),
    }


def _build_feature_sets(
    aligned: dict[str, Any],
    calibration_fraction: float,
    seed: int,
) -> dict[str, Any]:
    member_calibration_positions, member_test_positions = _split_positions(
        aligned["member_a"].shape[0],
        calibration_fraction,
        seed,
    )
    nonmember_calibration_positions, nonmember_test_positions = _split_positions(
        aligned["nonmember_a"].shape[0],
        calibration_fraction,
        seed + 1,
    )

    calibration_a_member = aligned["member_a"][member_calibration_positions]
    calibration_a_nonmember = aligned["nonmember_a"][nonmember_calibration_positions]
    calibration_b_member = aligned["member_b"][member_calibration_positions]
    calibration_b_nonmember = aligned["nonmember_b"][nonmember_calibration_positions]

    oriented_a_member, oriented_a_nonmember, orientation_a = _orient_memberness(
        calibration_a_member,
        calibration_a_nonmember,
    )
    oriented_b_member, oriented_b_nonmember, orientation_b = _orient_memberness(
        calibration_b_member,
        calibration_b_nonmember,
    )

    standardizer_a = _fit_standardizer(oriented_a_member, oriented_a_nonmember)
    standardizer_b = _fit_standardizer(oriented_b_member, oriented_b_nonmember)

    def transform_all(
        member_raw: np.ndarray,
        nonmember_raw: np.ndarray,
        orientation: str,
        standardizer: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        oriented_member = -member_raw if orientation == "negated" else member_raw
        oriented_nonmember = -nonmember_raw if orientation == "negated" else nonmember_raw
        return _apply_standardizer(oriented_member, standardizer), _apply_standardizer(oriented_nonmember, standardizer)

    calibration_a = np.concatenate(
        transform_all(calibration_a_member, calibration_a_nonmember, orientation_a, standardizer_a)
    )
    calibration_b = np.concatenate(
        transform_all(calibration_b_member, calibration_b_nonmember, orientation_b, standardizer_b)
    )

    test_a_member = aligned["member_a"][member_test_positions]
    test_a_nonmember = aligned["nonmember_a"][nonmember_test_positions]
    test_b_member = aligned["member_b"][member_test_positions]
    test_b_nonmember = aligned["nonmember_b"][nonmember_test_positions]

    test_a = np.concatenate(transform_all(test_a_member, test_a_nonmember, orientation_a, standardizer_a))
    test_b = np.concatenate(transform_all(test_b_member, test_b_nonmember, orientation_b, standardizer_b))

    calibration_labels = np.concatenate(
        [
            np.ones(member_calibration_positions.shape[0], dtype=int),
            np.zeros(nonmember_calibration_positions.shape[0], dtype=int),
        ]
    )
    test_labels = np.concatenate(
        [
            np.ones(member_test_positions.shape[0], dtype=int),
            np.zeros(nonmember_test_positions.shape[0], dtype=int),
        ]
    )

    return {
        "calibration": {
            "feature_a": calibration_a,
            "feature_b": calibration_b,
            "labels": calibration_labels,
        },
        "test": {
            "feature_a": test_a,
            "feature_b": test_b,
            "labels": test_labels,
        },
        "split": {
            "member_calibration_positions": member_calibration_positions.tolist(),
            "member_test_positions": member_test_positions.tolist(),
            "nonmember_calibration_positions": nonmember_calibration_positions.tolist(),
            "nonmember_test_positions": nonmember_test_positions.tolist(),
            "calibration_count_per_label": int(member_calibration_positions.shape[0])
            if member_calibration_positions.shape[0] == nonmember_calibration_positions.shape[0]
            else None,
            "test_count_per_label": int(member_test_positions.shape[0])
            if member_test_positions.shape[0] == nonmember_test_positions.shape[0]
            else None,
            "calibration_member_count": int(member_calibration_positions.shape[0]),
            "calibration_nonmember_count": int(nonmember_calibration_positions.shape[0]),
            "test_member_count": int(member_test_positions.shape[0]),
            "test_nonmember_count": int(nonmember_test_positions.shape[0]),
        },
        "orientations": {
            "surface_a": orientation_a,
            "surface_b": orientation_b,
        },
        "standardizers": {
            "surface_a": {key: _round6(value) for key, value in standardizer_a.items()},
            "surface_b": {key: _round6(value) for key, value in standardizer_b.items()},
        },
    }


def _evaluate_pairboard_once(
    *,
    aligned: dict[str, Any],
    surface_a_name: str,
    surface_b_name: str,
    calibration_fraction: float,
    seed: int,
    enable_tail_gated_cascade: bool = False,
    cascade_anchor_name: str | None = None,
    cascade_candidate_name: str = "logistic_2feature",
    cascade_route_fractions: list[float] | tuple[float, ...] | str | None = None,
    cascade_gammas: list[float] | tuple[float, ...] | str | None = None,
    cascade_secondary_cost_ratio: float = 0.25,
) -> dict[str, Any]:
    features = _build_feature_sets(aligned, calibration_fraction=calibration_fraction, seed=seed)

    calibration_candidate_scores = {
        surface_a_name: features["calibration"]["feature_a"],
        surface_b_name: features["calibration"]["feature_b"],
    }
    best_single_name, best_single_calibration_metrics = _choose_best_single(
        calibration_candidate_scores,
        features["calibration"]["labels"],
    )
    best_single_test_scores = (
        features["test"]["feature_a"] if best_single_name == surface_a_name else features["test"]["feature_b"]
    )

    weighted_calibration_scores, weighted_details = _weighted_average(
        calibration_features={
            surface_a_name: features["calibration"]["feature_a"],
            surface_b_name: features["calibration"]["feature_b"],
        },
        calibration_labels=features["calibration"]["labels"],
        evaluation_features={
            surface_a_name: features["calibration"]["feature_a"],
            surface_b_name: features["calibration"]["feature_b"],
        },
    )
    weighted_test_scores, _ = _weighted_average(
        calibration_features={
            surface_a_name: features["calibration"]["feature_a"],
            surface_b_name: features["calibration"]["feature_b"],
        },
        calibration_labels=features["calibration"]["labels"],
        evaluation_features={
            surface_a_name: features["test"]["feature_a"],
            surface_b_name: features["test"]["feature_b"],
        },
    )

    calibration_matrix = np.column_stack(
        [
            features["calibration"]["feature_a"],
            features["calibration"]["feature_b"],
        ]
    )
    test_matrix = np.column_stack(
        [
            features["test"]["feature_a"],
            features["test"]["feature_b"],
        ]
    )
    logistic_weights, logistic_bias = _fit_logistic_2feature(
        calibration_matrix,
        features["calibration"]["labels"],
    )
    logistic_calibration_scores = calibration_matrix @ logistic_weights + logistic_bias
    logistic_test_scores = test_matrix @ logistic_weights + logistic_bias

    sdn_calibration_scores, sdn_calibration_regions = _support_disconfirm_neutral(
        features["calibration"]["feature_a"],
        features["calibration"]["feature_b"],
    )
    sdn_test_scores, sdn_test_regions = _support_disconfirm_neutral(
        features["test"]["feature_a"],
        features["test"]["feature_b"],
    )

    calibration_candidates = {
        "best_single": best_single_calibration_metrics,
        "weighted_average": _metric_bundle(weighted_calibration_scores, features["calibration"]["labels"]),
        "logistic_2feature": _metric_bundle(logistic_calibration_scores, features["calibration"]["labels"]),
        "support_disconfirm_neutral": _metric_bundle(sdn_calibration_scores, features["calibration"]["labels"]),
    }
    test_candidates = {
        "best_single": _metric_bundle(best_single_test_scores, features["test"]["labels"]),
        "weighted_average": _metric_bundle(weighted_test_scores, features["test"]["labels"]),
        "logistic_2feature": _metric_bundle(logistic_test_scores, features["test"]["labels"]),
        "support_disconfirm_neutral": _metric_bundle(sdn_test_scores, features["test"]["labels"]),
    }
    analysis: dict[str, Any] = {
        "surface_orientations": features["orientations"],
        "surface_standardizers": features["standardizers"],
        "weighted_average": weighted_details,
        "logistic_2feature": {
            "weights": [
                _round6(logistic_weights[0]),
                _round6(logistic_weights[1]),
            ],
            "bias": _round6(logistic_bias),
        },
        "support_disconfirm_neutral": {
            "calibration_regions": sdn_calibration_regions,
            "test_regions": sdn_test_regions,
        },
    }

    if enable_tail_gated_cascade:
        resolved_anchor_name = cascade_anchor_name or surface_b_name
        calibration_candidates["tail_gated_cascade"], test_candidates["tail_gated_cascade"], cascade_analysis = (
            _build_tail_gated_cascade_candidate(
                calibration_scores={
                    surface_a_name: features["calibration"]["feature_a"],
                    surface_b_name: features["calibration"]["feature_b"],
                    "weighted_average": weighted_calibration_scores,
                    "logistic_2feature": logistic_calibration_scores,
                    "support_disconfirm_neutral": sdn_calibration_scores,
                },
                test_scores={
                    surface_a_name: features["test"]["feature_a"],
                    surface_b_name: features["test"]["feature_b"],
                    "weighted_average": weighted_test_scores,
                    "logistic_2feature": logistic_test_scores,
                    "support_disconfirm_neutral": sdn_test_scores,
                },
                calibration_labels=features["calibration"]["labels"],
                test_labels=features["test"]["labels"],
                anchor_name=resolved_anchor_name,
                candidate_name=cascade_candidate_name,
                surface_a_name=surface_a_name,
                surface_b_name=surface_b_name,
                route_fractions=cascade_route_fractions,
                gammas=cascade_gammas,
                secondary_cost_ratio=cascade_secondary_cost_ratio,
            )
        )
        analysis["tail_gated_cascade"] = cascade_analysis

    return {
        "seed": int(seed),
        "split": {
            "calibration_fraction": float(calibration_fraction),
            **features["split"],
        },
        "calibration": {
            "best_single": {
                "selected_surface": best_single_name,
                "selection_metric": "auc",
            },
            "candidates": calibration_candidates,
        },
        "test": {
            "best_single": {
                "selected_surface": best_single_name,
            },
            "candidates": test_candidates,
        },
        "analysis": analysis,
    }


def _series_summary(values: list[float]) -> dict[str, float]:
    series = np.asarray(values, dtype=float)
    return {
        "mean": _round6(float(series.mean())),
        "std": _round6(float(series.std())),
        "min": _round6(float(series.min())),
        "max": _round6(float(series.max())),
    }


def _summarize_repeated_holdout_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("Repeated holdout summary requires at least one run")

    candidate_names = list(runs[0]["test"]["candidates"].keys())
    selected_surface_counts: dict[str, int] = {}
    for run in runs:
        selected_surface = str(run["test"]["best_single"]["selected_surface"])
        selected_surface_counts[selected_surface] = selected_surface_counts.get(selected_surface, 0) + 1

    candidate_metrics: dict[str, dict[str, dict[str, float]]] = {}
    best_single_comparison: dict[str, dict[str, dict[str, float | int]]] = {}
    for candidate_name in candidate_names:
        candidate_metrics[candidate_name] = {}
        if candidate_name != "best_single":
            best_single_comparison[candidate_name] = {}
        for metric_name in PAIRBOARD_METRICS:
            values = [
                float(run["test"]["candidates"][candidate_name][metric_name])
                for run in runs
            ]
            candidate_metrics[candidate_name][metric_name] = _series_summary(values)

            if candidate_name == "best_single":
                continue

            deltas = np.asarray(
                [
                    float(run["test"]["candidates"][candidate_name][metric_name])
                    - float(run["test"]["candidates"]["best_single"][metric_name])
                    for run in runs
                ],
                dtype=float,
            )
            wins = int((deltas > 1e-12).sum())
            ties = int((np.abs(deltas) <= 1e-12).sum())
            losses = int((deltas < -1e-12).sum())
            best_single_comparison[candidate_name][metric_name] = {
                "win_count": wins,
                "tie_count": ties,
                "loss_count": losses,
                "mean_delta": _round6(float(deltas.mean())),
                "std_delta": _round6(float(deltas.std())),
                "min_delta": _round6(float(deltas.min())),
                "max_delta": _round6(float(deltas.max())),
            }

    return {
        "repeats": int(len(runs)),
        "selected_surface_counts": selected_surface_counts,
        "candidate_metrics": candidate_metrics,
        "best_single_comparison": best_single_comparison,
    }


def run_crossbox_pairboard(
    *,
    workspace: str | Path,
    surface_a_path: str | Path,
    surface_b_path: str | Path,
    surface_a_name: str | None = None,
    surface_b_name: str | None = None,
    surface_a_family: str | None = None,
    surface_b_family: str | None = None,
    calibration_fraction: float = 0.5,
    seed: int = 0,
    repeats: int = 1,
    enable_tail_gated_cascade: bool = False,
    cascade_anchor_name: str | None = None,
    cascade_candidate_name: str = "logistic_2feature",
    cascade_route_fractions: list[float] | tuple[float, ...] | str | None = None,
    cascade_gammas: list[float] | tuple[float, ...] | str | None = None,
    cascade_secondary_cost_ratio: float = 0.25,
) -> dict[str, Any]:
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    if int(repeats) < 1:
        raise ValueError("Pairboard repeated holdout requires repeats >= 1")

    surface_a = load_pairboard_surface(surface_a_path, name=surface_a_name, family=surface_a_family)
    surface_b = load_pairboard_surface(surface_b_path, name=surface_b_name, family=surface_b_family)
    aligned = _align_surfaces(surface_a, surface_b)
    primary = _evaluate_pairboard_once(
        aligned=aligned,
        surface_a_name=surface_a["name"],
        surface_b_name=surface_b["name"],
        calibration_fraction=calibration_fraction,
        seed=seed,
        enable_tail_gated_cascade=enable_tail_gated_cascade,
        cascade_anchor_name=cascade_anchor_name,
        cascade_candidate_name=cascade_candidate_name,
        cascade_route_fractions=cascade_route_fractions,
        cascade_gammas=cascade_gammas,
        cascade_secondary_cost_ratio=cascade_secondary_cost_ratio,
    )

    summary = {
        "status": "ready",
        "track": "cross-box",
        "method": "shared-score-pairboard",
        "mode": "crossbox-pairboard",
        "workspace": str(workspace_path),
        "workspace_name": workspace_path.name,
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
        },
        "surfaces": {
            "surface_a": {
                "name": surface_a["name"],
                "source_path": surface_a["source_path"],
                "source_kind": surface_a["source_kind"],
                "family": surface_a["family"],
            },
            "surface_b": {
                "name": surface_b["name"],
                "source_path": surface_b["source_path"],
                "source_kind": surface_b["source_kind"],
                "family": surface_b["family"],
            },
        },
        "pairboard": {
            "alignment": aligned["alignment"],
            "member_indices": aligned["member_indices"],
            "nonmember_indices": aligned["nonmember_indices"],
            "split": {
                "calibration_fraction": float(calibration_fraction),
                "seed": int(seed),
                "calibration_count_per_label": primary["split"]["calibration_count_per_label"],
                "test_count_per_label": primary["split"]["test_count_per_label"],
                "calibration_member_count": primary["split"]["calibration_member_count"],
                "calibration_nonmember_count": primary["split"]["calibration_nonmember_count"],
                "test_member_count": primary["split"]["test_member_count"],
                "test_nonmember_count": primary["split"]["test_nonmember_count"],
            },
        },
        "calibration": primary["calibration"],
        "test": primary["test"],
        "analysis": primary["analysis"],
        "notes": [
            "The pairboard reuses calibration-only orientation and scaling before evaluating the held-out test split.",
            "best_single is selected on calibration only, then transferred unchanged to the test board.",
            "This surface is generic across JSON score packets and GSA loss-score-export summaries.",
        ],
    }
    if enable_tail_gated_cascade:
        summary["notes"].append(
            "tail_gated_cascade reuses fold-local isotonic probabilities, anchors on one frozen surface threshold, and routes only the near-threshold band."
        )
    if int(repeats) > 1:
        repeated_runs: list[dict[str, Any]] = []
        for run_index in range(int(repeats)):
            run_seed = int(seed) + int(run_index) * REPEATED_HOLDOUT_SEED_STRIDE
            evaluation = primary if run_index == 0 else _evaluate_pairboard_once(
                aligned=aligned,
                surface_a_name=surface_a["name"],
                surface_b_name=surface_b["name"],
                calibration_fraction=calibration_fraction,
                seed=run_seed,
                enable_tail_gated_cascade=enable_tail_gated_cascade,
                cascade_anchor_name=cascade_anchor_name,
                cascade_candidate_name=cascade_candidate_name,
                cascade_route_fractions=cascade_route_fractions,
                cascade_gammas=cascade_gammas,
                cascade_secondary_cost_ratio=cascade_secondary_cost_ratio,
            )
            repeated_runs.append(
                {
                    "run_index": int(run_index),
                    "seed": int(run_seed),
                    "split": evaluation["split"],
                    "calibration": evaluation["calibration"],
                    "test": evaluation["test"],
                }
            )
        summary["repeated_holdout"] = {
            "configuration": {
                "repeats": int(repeats),
                "seed_start": int(seed),
                "seed_stride": REPEATED_HOLDOUT_SEED_STRIDE,
                "calibration_fraction": float(calibration_fraction),
                "protocol": "stratified-50-50-repeated-holdout"
                if abs(float(calibration_fraction) - 0.5) <= 1e-12
                else "stratified-repeated-holdout",
            },
            "runs": repeated_runs,
            "aggregate": _summarize_repeated_holdout_runs(repeated_runs),
            "notes": [
                "Each repeated run recalibrates orientation, scaling, and fusion weights on its own calibration half.",
                "best_single comparisons are computed against the transferred best_single candidate inside the same repeated run.",
            ],
        }
    (workspace_path / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary
