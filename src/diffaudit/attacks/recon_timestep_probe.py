from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def load_recon_vector_artifact(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    features = []
    labels = []
    for item in payload:
        feature = np.asarray(item[0], dtype=float)
        if feature.ndim != 1:
            raise ValueError("recon timestep probe expects 1D feature vectors per sample")
        features.append(feature)
        labels.append(int(item[1]))
    feature_array = np.asarray(features, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    return feature_array, label_array


def _threshold_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    tpr_at_1pct = 0.0
    for current_fpr, current_tpr in zip(fpr, tpr):
        if float(current_fpr) <= 0.01:
            tpr_at_1pct = max(tpr_at_1pct, float(current_tpr))
    return {
        "auc": auc,
        "tpr_at_1pct_fpr": float(tpr_at_1pct),
    }


def _orient_scores(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    if float(member_scores.mean()) < float(nonmember_scores.mean()):
        return -member_scores, -nonmember_scores, "negated"
    return member_scores, nonmember_scores, "identity"


def analyze_recon_timestep_probe(
    shadow_member: np.ndarray,
    shadow_nonmember: np.ndarray,
    target_member: np.ndarray,
    target_nonmember: np.ndarray,
) -> dict[str, object]:
    if shadow_member.shape[1] != shadow_nonmember.shape[1]:
        raise ValueError("shadow feature dimensions must match")
    if target_member.shape[1] != target_nonmember.shape[1]:
        raise ValueError("target feature dimensions must match")
    if shadow_member.shape[1] != target_member.shape[1]:
        raise ValueError("shadow and target feature dimensions must match")

    dim_count = int(shadow_member.shape[1])
    shadow_auc_per_dim = []
    target_auc_per_dim = []
    orientations = []
    for dim in range(dim_count):
        oriented_shadow_member, oriented_shadow_nonmember, orientation = _orient_scores(
            shadow_member[:, dim], shadow_nonmember[:, dim]
        )
        oriented_target_member, oriented_target_nonmember, _ = _orient_scores(
            target_member[:, dim], target_nonmember[:, dim]
        )
        shadow_labels = np.concatenate(
            [np.ones_like(oriented_shadow_member, dtype=int), np.zeros_like(oriented_shadow_nonmember, dtype=int)]
        )
        shadow_scores = np.concatenate([oriented_shadow_member, oriented_shadow_nonmember])
        target_labels = np.concatenate(
            [np.ones_like(oriented_target_member, dtype=int), np.zeros_like(oriented_target_nonmember, dtype=int)]
        )
        target_scores = np.concatenate([oriented_target_member, oriented_target_nonmember])
        shadow_auc_per_dim.append(float(roc_auc_score(shadow_labels, shadow_scores)))
        target_auc_per_dim.append(float(roc_auc_score(target_labels, target_scores)))
        orientations.append(orientation)

    best_dim = int(np.argmax(np.asarray(shadow_auc_per_dim, dtype=float)))
    baseline_dim = 0
    baseline_target_member, baseline_target_nonmember, baseline_orientation = _orient_scores(
        target_member[:, baseline_dim], target_nonmember[:, baseline_dim]
    )
    selected_target_member, selected_target_nonmember, _ = _orient_scores(
        target_member[:, best_dim], target_nonmember[:, best_dim]
    )
    target_labels = np.concatenate(
        [np.ones_like(selected_target_member, dtype=int), np.zeros_like(selected_target_nonmember, dtype=int)]
    )
    baseline_scores = np.concatenate([baseline_target_member, baseline_target_nonmember])
    selected_scores = np.concatenate([selected_target_member, selected_target_nonmember])

    baseline_metrics = _threshold_metrics(target_labels, baseline_scores)
    selected_metrics = _threshold_metrics(target_labels, selected_scores)

    return {
        "vector_length": dim_count,
        "baseline_dim": baseline_dim,
        "baseline_orientation": baseline_orientation,
        "best_shadow_dim": best_dim,
        "best_shadow_auc": round(float(shadow_auc_per_dim[best_dim]), 6),
        "best_shadow_orientation": orientations[best_dim],
        "baseline_target": {k: round(float(v), 6) for k, v in baseline_metrics.items()},
        "selected_target": {k: round(float(v), 6) for k, v in selected_metrics.items()},
        "target_auc_gain_vs_dim0": round(float(selected_metrics["auc"] - baseline_metrics["auc"]), 6),
        "target_top5_dims_by_shadow_auc": [
            {
                "dim": int(index),
                "shadow_auc": round(float(shadow_auc_per_dim[index]), 6),
                "target_auc": round(float(target_auc_per_dim[index]), 6),
                "orientation": orientations[index],
            }
            for index in np.argsort(np.asarray(shadow_auc_per_dim, dtype=float))[::-1][:5]
        ],
    }
