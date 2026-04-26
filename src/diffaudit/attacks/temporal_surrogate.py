"""Teacher-calibrated temporal surrogate scaffolding for 06-H1."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from diffaudit.attacks.crossbox_pairboard import load_pairboard_surface
from diffaudit.attacks.pia_adapter import (
    _build_pia_subset_loader,
    load_pia_model,
    prepare_pia_runtime,
    probe_pia_assets,
)
from diffaudit.attacks.tmiadm_adapter import (
    _alpha_bar_schedule,
    _apply_timestep_stride,
    _collect_eps_predictions_for_timesteps,
    _resolve_effective_timesteps,
)
from diffaudit.config import AuditConfig


def _round6(value: float) -> float:
    return round(float(value), 6)


def _window_groups(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.arange(count, dtype=int)
    groups = np.array_split(positions, 3)
    while len(groups) < 3:
        groups.append(groups[-1])
    early, mid, late = groups[0], groups[1], groups[2]
    return early, mid, late


def _mean_for_group(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return np.zeros(values.shape[0], dtype=float)
    return values[:, indices].mean(axis=1)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.maximum(np.abs(denominator), 1e-6)


def compute_temporal_feature_bank(
    *,
    timestep_predictions: torch.Tensor,
    timesteps: list[int] | tuple[int, ...],
) -> tuple[list[str], np.ndarray]:
    if timestep_predictions.ndim < 3:
        raise ValueError("Temporal surrogate feature bank expects [samples, timesteps, ...] predictions")
    if timestep_predictions.shape[1] < 3:
        raise ValueError("Temporal surrogate feature bank requires at least three timesteps")

    flattened = timestep_predictions.detach().cpu().numpy().reshape(
        timestep_predictions.shape[0],
        timestep_predictions.shape[1],
        -1,
    )
    per_timestep_abs = np.abs(flattened).mean(axis=2)
    per_timestep_sq = (flattened**2).mean(axis=2)
    x = np.asarray(list(timesteps), dtype=float)
    if x.shape[0] != per_timestep_abs.shape[1]:
        raise ValueError("Timesteps length must match prediction timestep axis")
    x_centered = x - float(x.mean())
    x_scale = max(float(np.std(x_centered)), 1.0)
    x_normalized = x_centered / x_scale

    early, mid, late = _window_groups(per_timestep_abs.shape[1])
    early_abs = _mean_for_group(per_timestep_abs, early)
    mid_abs = _mean_for_group(per_timestep_abs, mid)
    late_abs = _mean_for_group(per_timestep_abs, late)
    early_sq = _mean_for_group(per_timestep_sq, early)
    mid_sq = _mean_for_group(per_timestep_sq, mid)
    late_sq = _mean_for_group(per_timestep_sq, late)

    global_slope = np.zeros(per_timestep_abs.shape[0], dtype=float)
    late_slope = np.zeros(per_timestep_abs.shape[0], dtype=float)
    curvature = np.zeros(per_timestep_abs.shape[0], dtype=float)
    abs_std = per_timestep_abs.std(axis=1)
    abs_iqr = np.percentile(per_timestep_abs, 75, axis=1) - np.percentile(per_timestep_abs, 25, axis=1)
    abs_max_jump = np.abs(np.diff(per_timestep_abs, axis=1)).max(axis=1)

    late_x = x_normalized[late]
    for row_index in range(per_timestep_abs.shape[0]):
        global_slope[row_index] = float(np.polyfit(x_normalized, per_timestep_abs[row_index], deg=1)[0])
        if late_x.size >= 2:
            late_slope[row_index] = float(np.polyfit(late_x, per_timestep_abs[row_index, late], deg=1)[0])
        else:
            late_slope[row_index] = 0.0
        curvature[row_index] = float(np.polyfit(x_normalized, per_timestep_abs[row_index], deg=2)[0])

    feature_names = [
        "eps_abs_mean_early",
        "eps_abs_mean_mid",
        "eps_abs_mean_late",
        "eps_sq_mean_early",
        "eps_sq_mean_mid",
        "eps_sq_mean_late",
        "eps_abs_late_minus_early",
        "eps_abs_late_over_early",
        "eps_abs_global_slope",
        "eps_abs_late_slope",
        "eps_abs_curvature",
        "eps_abs_std",
        "eps_abs_iqr",
        "eps_abs_max_jump",
    ]
    features = np.column_stack(
        [
            early_abs,
            mid_abs,
            late_abs,
            early_sq,
            mid_sq,
            late_sq,
            late_abs - early_abs,
            _safe_ratio(late_abs, early_abs),
            global_slope,
            late_slope,
            curvature,
            abs_std,
            abs_iqr,
            abs_max_jump,
        ]
    )
    return feature_names, features.astype(float)


def _feature_packet_payload(
    *,
    feature_names: list[str],
    timesteps: list[int],
    member_indices: list[int],
    nonmember_indices: list[int],
    member_features: np.ndarray,
    nonmember_features: np.ndarray,
) -> dict[str, Any]:
    return {
        "feature_suite": "temporal-surrogate-v1",
        "feature_names": feature_names,
        "timesteps": [int(value) for value in timesteps],
        "member_indices": [int(value) for value in member_indices],
        "nonmember_indices": [int(value) for value in nonmember_indices],
        "member_features": [[_round6(value) for value in row] for row in member_features.tolist()],
        "nonmember_features": [[_round6(value) for value in row] for row in nonmember_features.tolist()],
    }


def export_temporal_surrogate_feature_packet(
    config: AuditConfig,
    workspace: str | Path,
    repo_root: str | Path = "external/PIA",
    member_split_root: str | Path | None = None,
    device: str = "cpu",
    max_samples: int | None = None,
    batch_size: int = 8,
    scan_timesteps: list[int] | tuple[int, ...] | None = None,
    noise_seed: int = 0,
    timestep_jitter_radius: int = 0,
    timestep_stride: int = 1,
    provenance_status: str = "workspace-verified",
) -> dict[str, Any]:
    started_at = time.perf_counter()
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    runtime_context = prepare_pia_runtime(
        config,
        repo_root=repo_root,
        member_split_root=member_split_root,
        device=device,
    )
    model, weights_key = load_pia_model(
        runtime_context.checkpoint_path,
        runtime_context.model_module,
        device=device,
    )
    split_root = Path(member_split_root) if member_split_root else Path(repo_root) / "DDPM"
    asset_summary = probe_pia_assets(
        dataset=runtime_context.plan.dataset,
        dataset_root=runtime_context.plan.data_root,
        model_dir=runtime_context.plan.model_dir,
        member_split_root=split_root,
    )
    selected_max_samples = max_samples or runtime_context.plan.num_samples
    member_loader, member_indices = _build_pia_subset_loader(
        dataset=runtime_context.plan.dataset,
        dataset_dir=asset_summary["paths"]["dataset_dir"],
        member_split_path=asset_summary["paths"]["member_split"],
        batch_size=batch_size,
        max_samples=selected_max_samples,
        membership="member",
    )
    nonmember_loader, nonmember_indices = _build_pia_subset_loader(
        dataset=runtime_context.plan.dataset,
        dataset_dir=asset_summary["paths"]["dataset_dir"],
        member_split_path=asset_summary["paths"]["member_split"],
        batch_size=batch_size,
        max_samples=selected_max_samples,
        membership="nonmember",
    )

    timesteps = [int(value) for value in (scan_timesteps or [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240])]
    effective_timesteps = _resolve_effective_timesteps(
        timesteps,
        timestep_jitter_radius=timestep_jitter_radius,
        noise_seed=noise_seed,
    )
    effective_timesteps = _apply_timestep_stride(
        effective_timesteps,
        timestep_stride=timestep_stride,
    )
    alpha_bars = _alpha_bar_schedule()
    member_predictions = _collect_eps_predictions_for_timesteps(
        model=model,
        loader=member_loader,
        timesteps=effective_timesteps,
        alpha_bars=alpha_bars,
        device=device,
        noise_seed=noise_seed,
    )
    nonmember_predictions = _collect_eps_predictions_for_timesteps(
        model=model,
        loader=nonmember_loader,
        timesteps=effective_timesteps,
        alpha_bars=alpha_bars,
        device=device,
        noise_seed=noise_seed,
    )
    feature_names, member_features = compute_temporal_feature_bank(
        timestep_predictions=member_predictions,
        timesteps=effective_timesteps,
    )
    _, nonmember_features = compute_temporal_feature_bank(
        timestep_predictions=nonmember_predictions,
        timesteps=effective_timesteps,
    )

    feature_packet = _feature_packet_payload(
        feature_names=feature_names,
        timesteps=effective_timesteps,
        member_indices=member_indices,
        nonmember_indices=nonmember_indices,
        member_features=member_features,
        nonmember_features=nonmember_features,
    )
    feature_packet_path = workspace_path / "feature-packet.json"
    feature_packet_path.write_text(
        json.dumps(feature_packet, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    result = {
        "status": "ready",
        "track": "gray-box",
        "method": "temporal-surrogate",
        "mode": "feature-packet-export",
        "workspace": str(workspace_path),
        "workspace_name": workspace_path.name,
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
            "feature_packet": str(feature_packet_path),
        },
        "checks": {
            **asset_summary["checks"],
            "runtime_probe_ready": True,
            "feature_packet_ready": True,
        },
        "runtime": {
            "repo_root": str(Path(repo_root)),
            "model_dir": runtime_context.plan.model_dir,
            "dataset_root": runtime_context.plan.data_root,
            "member_split_root": str(split_root),
            "batch_size": int(batch_size),
            "max_samples": int(selected_max_samples),
            "num_samples": int(len(member_indices)),
            "weights_key": weights_key,
            "scan_timesteps": timesteps,
            "effective_timesteps": effective_timesteps,
            "noise_seed": int(noise_seed),
            "elapsed_seconds": round(time.perf_counter() - started_at, 6),
        },
        "feature_bank": {
            "suite": "temporal-surrogate-v1",
            "feature_names": feature_names,
            "feature_count": int(len(feature_names)),
        },
        "sample_count_per_split": int(len(member_indices)),
        "provenance_status": provenance_status,
        "notes": [
            "This packet exports target-only temporal summary features for 06-H1 teacher-calibrated surrogate scoping.",
            "The packet does not fit or calibrate a surrogate by itself.",
        ],
    }
    (workspace_path / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return result


def _load_feature_packet(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "path": str(Path(path)),
        "feature_names": [str(value) for value in payload["feature_names"]],
        "member_indices": [int(value) for value in payload.get("member_indices", [])]
        if payload.get("member_indices") is not None
        else None,
        "nonmember_indices": [int(value) for value in payload.get("nonmember_indices", [])]
        if payload.get("nonmember_indices") is not None
        else None,
        "member_features": np.asarray(payload["member_features"], dtype=float),
        "nonmember_features": np.asarray(payload["nonmember_features"], dtype=float),
    }


def _align_feature_split(
    *,
    features: np.ndarray,
    feature_indices: list[int] | None,
    scores: np.ndarray,
    score_indices: list[int] | None,
) -> tuple[np.ndarray, np.ndarray, list[int] | None, str]:
    if feature_indices is not None and score_indices is not None:
        score_lookup = {int(index): position for position, index in enumerate(score_indices)}
        shared_indices = [int(index) for index in feature_indices if int(index) in score_lookup]
        if not shared_indices:
            raise ValueError("Feature packet and teacher surface do not share any indices")
        feature_positions = [position for position, index in enumerate(feature_indices) if int(index) in score_lookup]
        score_positions = [score_lookup[index] for index in shared_indices]
        return (
            features[np.asarray(feature_positions, dtype=int)],
            scores[np.asarray(score_positions, dtype=int)],
            shared_indices,
            "shared-index-intersection",
        )

    if features.shape[0] != scores.shape[0]:
        raise ValueError("Feature packet and teacher surface must have equal per-split lengths when indices are absent")
    retained_indices = feature_indices if feature_indices is not None else score_indices
    strategy = "positional-order" if retained_indices is None else "positional-order-single-index"
    return features.copy(), scores.copy(), retained_indices, strategy


def _aligned_teacher_board(
    *,
    feature_packet: dict[str, Any],
    teacher_surface: dict[str, Any],
) -> dict[str, Any]:
    member_features, member_scores, member_indices, member_strategy = _align_feature_split(
        features=feature_packet["member_features"],
        feature_indices=feature_packet["member_indices"],
        scores=teacher_surface["member_scores"],
        score_indices=teacher_surface["member_indices"],
    )
    nonmember_features, nonmember_scores, nonmember_indices, nonmember_strategy = _align_feature_split(
        features=feature_packet["nonmember_features"],
        feature_indices=feature_packet["nonmember_indices"],
        scores=teacher_surface["nonmember_scores"],
        score_indices=teacher_surface["nonmember_indices"],
    )
    return {
        "feature_names": feature_packet["feature_names"],
        "member_features": member_features,
        "nonmember_features": nonmember_features,
        "member_scores": member_scores,
        "nonmember_scores": nonmember_scores,
        "member_indices": member_indices,
        "nonmember_indices": nonmember_indices,
        "alignment": {
            "member_strategy": member_strategy,
            "nonmember_strategy": nonmember_strategy,
            "shared_member_count": int(member_features.shape[0]),
            "shared_nonmember_count": int(nonmember_features.shape[0]),
        },
    }


def _build_teacher_arrays(aligned: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.concatenate([aligned["member_features"], aligned["nonmember_features"]], axis=0)
    teacher_scores = np.concatenate([aligned["member_scores"], aligned["nonmember_scores"]], axis=0)
    labels = np.concatenate(
        [
            np.ones(aligned["member_features"].shape[0], dtype=int),
            np.zeros(aligned["nonmember_features"].shape[0], dtype=int),
        ]
    )
    return features, teacher_scores, labels


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


def _best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    sorted_scores = np.sort(scores)
    thresholds = np.r_[
        sorted_scores[0] - 1e-6,
        (sorted_scores[:-1] + sorted_scores[1:]) / 2.0,
        sorted_scores[-1] + 1e-6,
    ]
    best_threshold = float(thresholds[0])
    best_accuracy = -1.0
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        accuracy = float((predictions == labels).mean())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return best_threshold


def _accuracy_best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    threshold = _best_threshold(scores, labels)
    predictions = (scores >= threshold).astype(int)
    return float((predictions == labels).mean())


def _tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    fpr, tpr = _roc_curve_points(scores, labels)
    valid = tpr[fpr <= float(target_fpr)]
    if valid.size == 0:
        return 0.0
    return float(valid.max())


def _orient_memberness(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, str]:
    member_mean = float(scores[labels == 1].mean())
    nonmember_mean = float(scores[labels == 0].mean())
    if member_mean < nonmember_mean:
        return -scores, "negated"
    return scores.copy(), "identity"


def _membership_metric_bundle(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    oriented_scores, _ = _orient_memberness(scores, labels)
    return {
        "auc": _round6(_auc_score(oriented_scores, labels)),
        "asr": _round6(_accuracy_best_threshold(oriented_scores, labels)),
        "tpr_at_1pct_fpr": _round6(_tpr_at_fpr(oriented_scores, labels, 0.01)),
        "tpr_at_0_1pct_fpr": _round6(_tpr_at_fpr(oriented_scores, labels, 0.001)),
    }


def _fit_bagged_quantile_ensemble(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    quantiles: Iterable[float],
    bag_count: int,
    l2_alpha: float,
    random_seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(random_seed))
    quantile_list = [float(value) for value in quantiles]
    ensemble: list[dict[str, Any]] = []
    for bag_index in range(int(bag_count)):
        quantile = quantile_list[bag_index % len(quantile_list)]
        bootstrap_indices = rng.integers(0, features.shape[0], size=features.shape[0])
        bootstrap_features = features[bootstrap_indices]
        bootstrap_targets = targets[bootstrap_indices]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(bootstrap_features)
        model = QuantileRegressor(
            quantile=quantile,
            alpha=float(l2_alpha),
            solver="highs",
        )
        model.fit(scaled_features, bootstrap_targets)
        ensemble.append(
            {
                "quantile": quantile,
                "scaler": scaler,
                "model": model,
            }
        )
    return ensemble


def _predict_bagged_quantile_ensemble(ensemble: list[dict[str, Any]], features: np.ndarray) -> np.ndarray:
    predictions = []
    for head in ensemble:
        scaled = head["scaler"].transform(features)
        predictions.append(head["model"].predict(scaled))
    return np.mean(np.vstack(predictions), axis=0)


def _threshold_cv(thresholds: list[float]) -> float:
    threshold_array = np.asarray(thresholds, dtype=float)
    mean_abs = max(float(np.abs(threshold_array).mean()), 1e-6)
    return float(threshold_array.std(ddof=0) / mean_abs)


def _safe_corr(value: float) -> float:
    if np.isnan(value):
        return 0.0
    return float(value)


def _evaluate_teacher_oof(
    *,
    features: np.ndarray,
    teacher_scores: np.ndarray,
    labels: np.ndarray,
    quantiles: Iterable[float],
    bag_count: int,
    l2_alpha: float,
    cv_splits: int,
    cv_repeats: int,
    random_seed: int,
) -> dict[str, Any]:
    splitter = RepeatedStratifiedKFold(
        n_splits=int(cv_splits),
        n_repeats=int(cv_repeats),
        random_state=int(random_seed),
    )
    prediction_sum = np.zeros(features.shape[0], dtype=float)
    prediction_count = np.zeros(features.shape[0], dtype=int)
    thresholds: list[float] = []

    for fold_index, (train_indices, test_indices) in enumerate(splitter.split(features, labels)):
        ensemble = _fit_bagged_quantile_ensemble(
            features=features[train_indices],
            targets=teacher_scores[train_indices],
            quantiles=quantiles,
            bag_count=bag_count,
            l2_alpha=l2_alpha,
            random_seed=int(random_seed) + int(fold_index),
        )
        train_raw = _predict_bagged_quantile_ensemble(ensemble, features[train_indices])
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(train_raw, teacher_scores[train_indices])
        train_calibrated = calibrator.predict(train_raw)
        oriented_train, _ = _orient_memberness(train_calibrated, labels[train_indices])
        thresholds.append(_best_threshold(oriented_train, labels[train_indices]))

        test_raw = _predict_bagged_quantile_ensemble(ensemble, features[test_indices])
        test_calibrated = calibrator.predict(test_raw)
        prediction_sum[test_indices] += test_calibrated
        prediction_count[test_indices] += 1

    if np.any(prediction_count == 0):
        raise RuntimeError("Temporal surrogate OOF evaluation left some samples without predictions")
    oof_predictions = prediction_sum / prediction_count
    teacher_metrics = _membership_metric_bundle(teacher_scores, labels)
    surrogate_metrics = _membership_metric_bundle(oof_predictions, labels)
    spearman = _safe_corr(float(spearmanr(oof_predictions, teacher_scores).statistic))
    pearson = _safe_corr(float(pearsonr(oof_predictions, teacher_scores).statistic))
    threshold_cv = _threshold_cv(thresholds)

    return {
        "oof_predictions": oof_predictions,
        "teacher_metrics": teacher_metrics,
        "surrogate_metrics": surrogate_metrics,
        "metrics": {
            "spearman": _round6(spearman),
            "pearson": _round6(pearson),
            "auc": surrogate_metrics["auc"],
            "asr": surrogate_metrics["asr"],
            "tpr_at_1pct_fpr": surrogate_metrics["tpr_at_1pct_fpr"],
            "tpr_at_0_1pct_fpr": surrogate_metrics["tpr_at_0_1pct_fpr"],
            "teacher_auc": teacher_metrics["auc"],
            "teacher_tpr_at_1pct_fpr": teacher_metrics["tpr_at_1pct_fpr"],
            "teacher_tpr_at_0_1pct_fpr": teacher_metrics["tpr_at_0_1pct_fpr"],
            "auc_delta_vs_teacher": _round6(surrogate_metrics["auc"] - teacher_metrics["auc"]),
            "tpr_at_1pct_fpr_delta_vs_teacher": _round6(
                surrogate_metrics["tpr_at_1pct_fpr"] - teacher_metrics["tpr_at_1pct_fpr"]
            ),
            "tpr_at_0_1pct_fpr_delta_vs_teacher": _round6(
                surrogate_metrics["tpr_at_0_1pct_fpr"] - teacher_metrics["tpr_at_0_1pct_fpr"]
            ),
            "threshold_cv": _round6(threshold_cv),
        },
        "thresholds": thresholds,
        "hard_gates": {
            "spearman_ge_0_8": bool(spearman >= 0.8),
            "pearson_ge_0_8": bool(pearson >= 0.8),
            "auc_delta_abs_le_0_05": bool(abs(surrogate_metrics["auc"] - teacher_metrics["auc"]) <= 0.05),
            "tpr_at_1pct_fpr_delta_abs_le_0_05": bool(
                abs(surrogate_metrics["tpr_at_1pct_fpr"] - teacher_metrics["tpr_at_1pct_fpr"]) <= 0.05
            ),
            "threshold_cv_lt_0_15": bool(threshold_cv < 0.15),
        },
    }


def _fit_full_teacher_model(
    *,
    features: np.ndarray,
    teacher_scores: np.ndarray,
    quantiles: Iterable[float],
    bag_count: int,
    l2_alpha: float,
    random_seed: int,
) -> dict[str, Any]:
    ensemble = _fit_bagged_quantile_ensemble(
        features=features,
        targets=teacher_scores,
        quantiles=quantiles,
        bag_count=bag_count,
        l2_alpha=l2_alpha,
        random_seed=random_seed,
    )
    raw_predictions = _predict_bagged_quantile_ensemble(ensemble, features)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_predictions, teacher_scores)
    return {
        "ensemble": ensemble,
        "calibrator": calibrator,
    }


def _predict_with_full_teacher_model(model_bundle: dict[str, Any], features: np.ndarray) -> np.ndarray:
    raw_predictions = _predict_bagged_quantile_ensemble(model_bundle["ensemble"], features)
    return model_bundle["calibrator"].predict(raw_predictions)


def _feature_shift_audit(
    teacher_features: np.ndarray,
    transfer_features: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    teacher_mean = teacher_features.mean(axis=0)
    transfer_mean = transfer_features.mean(axis=0)
    teacher_std = np.maximum(teacher_features.std(axis=0, ddof=0), 1e-6)
    mean_shift_z = np.abs(transfer_mean - teacher_mean) / teacher_std
    max_index = int(np.argmax(mean_shift_z))
    return {
        "max_feature_mean_shift_z": _round6(mean_shift_z[max_index]),
        "max_feature_name": feature_names[max_index],
        "per_feature_mean_shift_z": {
            feature_names[index]: _round6(value)
            for index, value in enumerate(mean_shift_z.tolist())
        },
    }


def evaluate_temporal_surrogate_packets(
    *,
    workspace: str | Path,
    teacher_feature_packet: str | Path,
    teacher_score_surface: str | Path,
    teacher_score_family: str | None = None,
    transfer_feature_packet: str | Path | None = None,
    quantiles: Iterable[float] = (0.2, 0.35, 0.5, 0.65, 0.8),
    bag_count: int = 8,
    l2_alpha: float = 0.01,
    cv_splits: int = 4,
    cv_repeats: int = 2,
    random_seed: int = 0,
    provenance_status: str = "workspace-verified",
) -> dict[str, Any]:
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    teacher_packet = _load_feature_packet(teacher_feature_packet)
    teacher_surface = load_pairboard_surface(
        teacher_score_surface,
        name="teacher_surface",
        family=teacher_score_family,
    )
    aligned = _aligned_teacher_board(
        feature_packet=teacher_packet,
        teacher_surface=teacher_surface,
    )
    teacher_features, teacher_scores, teacher_labels = _build_teacher_arrays(aligned)
    teacher_oof = _evaluate_teacher_oof(
        features=teacher_features,
        teacher_scores=teacher_scores,
        labels=teacher_labels,
        quantiles=quantiles,
        bag_count=bag_count,
        l2_alpha=l2_alpha,
        cv_splits=cv_splits,
        cv_repeats=cv_repeats,
        random_seed=random_seed,
    )
    full_model = _fit_full_teacher_model(
        features=teacher_features,
        teacher_scores=teacher_scores,
        quantiles=quantiles,
        bag_count=bag_count,
        l2_alpha=l2_alpha,
        random_seed=random_seed,
    )

    result = {
        "status": "ready",
        "track": "gray-box",
        "method": "temporal-surrogate",
        "mode": "teacher-calibrated-temporal-surrogate",
        "workspace": str(workspace_path),
        "workspace_name": workspace_path.name,
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
        },
        "model": {
            "bag_count": int(bag_count),
            "quantiles": [_round6(value) for value in quantiles],
            "l2_alpha": _round6(l2_alpha),
            "cv_splits": int(cv_splits),
            "cv_repeats": int(cv_repeats),
            "random_seed": int(random_seed),
        },
        "teacher": {
            "feature_packet": str(Path(teacher_feature_packet)),
            "score_surface": str(Path(teacher_score_surface)),
            "score_family": teacher_score_family,
            "alignment": aligned["alignment"],
            "member_indices": aligned["member_indices"],
            "nonmember_indices": aligned["nonmember_indices"],
            "sample_count_per_split": int(aligned["member_features"].shape[0]),
            "metrics": teacher_oof["metrics"],
            "hard_gates": teacher_oof["hard_gates"],
        },
        "provenance_status": provenance_status,
        "notes": [
            "The surrogate is trained only on the teacher packet and remains frozen before any transfer packet is scored.",
            "OOF metrics are reported on teacher alignment first; transfer metrics must not trigger retraining.",
        ],
    }

    if transfer_feature_packet is not None:
        transfer_packet = _load_feature_packet(transfer_feature_packet)
        if transfer_packet["feature_names"] != teacher_packet["feature_names"]:
            raise ValueError("Transfer packet feature names must match the teacher packet exactly")
        transfer_features = np.concatenate(
            [transfer_packet["member_features"], transfer_packet["nonmember_features"]],
            axis=0,
        )
        transfer_labels = np.concatenate(
            [
                np.ones(transfer_packet["member_features"].shape[0], dtype=int),
                np.zeros(transfer_packet["nonmember_features"].shape[0], dtype=int),
            ]
        )
        transfer_predictions = _predict_with_full_teacher_model(full_model, transfer_features)
        transfer_metrics = _membership_metric_bundle(transfer_predictions, transfer_labels)
        result["transfer"] = {
            "feature_packet": str(Path(transfer_feature_packet)),
            "sample_count_per_split": int(transfer_packet["member_features"].shape[0]),
            "metrics": transfer_metrics,
            "shift_audit": _feature_shift_audit(
                teacher_features=teacher_features,
                transfer_features=transfer_features,
                feature_names=teacher_packet["feature_names"],
            ),
        }

    (workspace_path / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return result
