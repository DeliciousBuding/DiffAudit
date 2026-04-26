"""Temporal likelihood-ratio fallback evaluator for 06-H2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from diffaudit.attacks.temporal_surrogate import (
    _feature_shift_audit,
    _membership_metric_bundle,
    _round6,
    _threshold_cv,
    _load_feature_packet,
)


def _fit_univariate_gaussian(values: np.ndarray, variance_floor: float = 1e-6) -> dict[str, float]:
    variance = max(float(values.var(ddof=0)), float(variance_floor))
    return {
        "mean": float(values.mean()),
        "variance": variance,
    }


def _gaussian_log_likelihood_ratio(
    values: np.ndarray,
    *,
    member_fit: dict[str, float],
    nonmember_fit: dict[str, float],
) -> np.ndarray:
    member_mean = float(member_fit["mean"])
    member_var = float(member_fit["variance"])
    nonmember_mean = float(nonmember_fit["mean"])
    nonmember_var = float(nonmember_fit["variance"])
    member_logpdf = -0.5 * np.log(2.0 * np.pi * member_var) - ((values - member_mean) ** 2) / (2.0 * member_var)
    nonmember_logpdf = -0.5 * np.log(2.0 * np.pi * nonmember_var) - (
        (values - nonmember_mean) ** 2
    ) / (2.0 * nonmember_var)
    return member_logpdf - nonmember_logpdf


def _get_feature_values(packet: dict[str, Any], feature_name: str) -> tuple[np.ndarray, np.ndarray]:
    feature_names = packet["feature_names"]
    if feature_name not in feature_names:
        raise ValueError(f"Requested temporal LR feature '{feature_name}' was not found in feature packet")
    feature_index = feature_names.index(feature_name)
    return (
        packet["member_features"][:, feature_index].astype(float),
        packet["nonmember_features"][:, feature_index].astype(float),
    )


def _evaluate_feature_candidate(
    *,
    member_values: np.ndarray,
    nonmember_values: np.ndarray,
    cv_splits: int,
    cv_repeats: int,
    random_seed: int,
) -> dict[str, Any]:
    values = np.concatenate([member_values, nonmember_values], axis=0)
    labels = np.concatenate(
        [
            np.ones(member_values.shape[0], dtype=int),
            np.zeros(nonmember_values.shape[0], dtype=int),
        ]
    )
    splitter = RepeatedStratifiedKFold(
        n_splits=int(cv_splits),
        n_repeats=int(cv_repeats),
        random_state=int(random_seed),
    )
    prediction_sum = np.zeros(values.shape[0], dtype=float)
    prediction_count = np.zeros(values.shape[0], dtype=int)
    thresholds: list[float] = []

    for train_indices, test_indices in splitter.split(values.reshape(-1, 1), labels):
        train_values = values[train_indices]
        train_labels = labels[train_indices]
        member_fit = _fit_univariate_gaussian(train_values[train_labels == 1])
        nonmember_fit = _fit_univariate_gaussian(train_values[train_labels == 0])
        train_scores = _gaussian_log_likelihood_ratio(
            train_values,
            member_fit=member_fit,
            nonmember_fit=nonmember_fit,
        )
        from diffaudit.attacks.temporal_surrogate import _best_threshold

        thresholds.append(_best_threshold(train_scores, train_labels))
        test_scores = _gaussian_log_likelihood_ratio(
            values[test_indices],
            member_fit=member_fit,
            nonmember_fit=nonmember_fit,
        )
        prediction_sum[test_indices] += test_scores
        prediction_count[test_indices] += 1

    if np.any(prediction_count == 0):
        raise RuntimeError("Temporal LR evaluation left some samples without OOF predictions")
    oof_scores = prediction_sum / prediction_count
    metrics = _membership_metric_bundle(oof_scores, labels)
    member_fit = _fit_univariate_gaussian(member_values)
    nonmember_fit = _fit_univariate_gaussian(nonmember_values)
    return {
        "metrics": {
            **metrics,
            "threshold_cv": _round6(_threshold_cv(thresholds)),
            "member_value_mean": _round6(member_fit["mean"]),
            "nonmember_value_mean": _round6(nonmember_fit["mean"]),
        },
        "density_fit": {
            "member": {
                "distribution": "gaussian",
                "mean": _round6(member_fit["mean"]),
                "variance": _round6(member_fit["variance"]),
            },
            "nonmember": {
                "distribution": "gaussian",
                "mean": _round6(nonmember_fit["mean"]),
                "variance": _round6(nonmember_fit["variance"]),
            },
        },
        "full_fit": {
            "member": member_fit,
            "nonmember": nonmember_fit,
        },
    }


def evaluate_temporal_lr_packets(
    *,
    workspace: str | Path,
    calibration_feature_packet: str | Path,
    transfer_feature_packet: str | Path | None = None,
    primary_candidate: str = "eps_abs_mean_late",
    sensitivity_candidate: str = "eps_abs_late_over_early",
    cv_splits: int = 4,
    cv_repeats: int = 2,
    random_seed: int = 0,
    provenance_status: str = "workspace-verified",
) -> dict[str, Any]:
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    calibration_packet = _load_feature_packet(calibration_feature_packet)
    candidate_names = [primary_candidate]
    if sensitivity_candidate != primary_candidate:
        candidate_names.append(sensitivity_candidate)

    candidates: dict[str, Any] = {}
    chosen_full_fit: dict[str, Any] | None = None
    for candidate_name in candidate_names:
        member_values, nonmember_values = _get_feature_values(calibration_packet, candidate_name)
        candidate_result = _evaluate_feature_candidate(
            member_values=member_values,
            nonmember_values=nonmember_values,
            cv_splits=cv_splits,
            cv_repeats=cv_repeats,
            random_seed=random_seed,
        )
        candidates[candidate_name] = {
            "metrics": candidate_result["metrics"],
            "density_fit": candidate_result["density_fit"],
        }
        if candidate_name == primary_candidate:
            chosen_full_fit = candidate_result["full_fit"]

    if chosen_full_fit is None:
        raise RuntimeError("Temporal LR evaluation could not fit the primary candidate")

    result = {
        "status": "ready",
        "track": "gray-box",
        "method": "temporal-lr",
        "mode": "temporal-lr-calibration-transfer",
        "workspace": str(workspace_path),
        "workspace_name": workspace_path.name,
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
        },
        "calibration": {
            "feature_packet": str(Path(calibration_feature_packet)),
            "sample_count_per_split": int(calibration_packet["member_features"].shape[0]),
            "primary_candidate": primary_candidate,
            "sensitivity_candidate": sensitivity_candidate,
            "candidates": candidates,
        },
        "provenance_status": provenance_status,
        "notes": [
            "H2 keeps the fallback surface simple: fixed late-window mean as the primary temporal statistic, early/late ratio as sensitivity.",
            "The transfer packet reuses the frozen calibration-side Gaussian LR fit without refitting.",
        ],
    }

    if transfer_feature_packet is not None:
        transfer_packet = _load_feature_packet(transfer_feature_packet)
        if transfer_packet["feature_names"] != calibration_packet["feature_names"]:
            raise ValueError("Transfer packet feature names must match the calibration packet exactly")
        member_values, nonmember_values = _get_feature_values(transfer_packet, primary_candidate)
        transfer_scores = _gaussian_log_likelihood_ratio(
            np.concatenate([member_values, nonmember_values], axis=0),
            member_fit=chosen_full_fit["member"],
            nonmember_fit=chosen_full_fit["nonmember"],
        )
        transfer_labels = np.concatenate(
            [
                np.ones(member_values.shape[0], dtype=int),
                np.zeros(nonmember_values.shape[0], dtype=int),
            ]
        )
        transfer_member_features = transfer_packet["member_features"]
        transfer_nonmember_features = transfer_packet["nonmember_features"]
        calibration_features = np.concatenate(
            [calibration_packet["member_features"], calibration_packet["nonmember_features"]],
            axis=0,
        )
        transfer_features = np.concatenate([transfer_member_features, transfer_nonmember_features], axis=0)
        result["transfer"] = {
            "feature_packet": str(Path(transfer_feature_packet)),
            "primary_candidate": primary_candidate,
            "sample_count_per_split": int(transfer_member_features.shape[0]),
            "metrics": _membership_metric_bundle(transfer_scores, transfer_labels),
            "shift_audit": _feature_shift_audit(
                teacher_features=calibration_features,
                transfer_features=transfer_features,
                feature_names=calibration_packet["feature_names"],
            ),
        }

    (workspace_path / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return result
