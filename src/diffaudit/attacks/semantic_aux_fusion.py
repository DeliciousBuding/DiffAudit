from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def load_semantic_aux_records(path: str | Path) -> list[dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("semantic-aux records payload must be a list")
    return payload


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    rank_a = _rankdata(values_a)
    rank_b = _rankdata(values_b)
    std_a = rank_a.std(ddof=0)
    std_b = rank_b.std(ddof=0)
    if std_a == 0.0 or std_b == 0.0:
        return 0.0
    corr = np.corrcoef(rank_a, rank_b)[0, 1]
    return float(corr)


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(values.std(ddof=0))
    if std == 0.0:
        return values - float(values.mean())
    return (values - float(values.mean())) / std


def _orient_memberness(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    member_mean = float(values[labels == 1].mean())
    nonmember_mean = float(values[labels == 0].mean())
    if member_mean < nonmember_mean:
        return -values
    return values


def _threshold_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    tpr_at_1pct = 0.0
    for current_fpr, current_tpr in zip(fpr, tpr):
        if float(current_fpr) <= 0.01:
            tpr_at_1pct = max(tpr_at_1pct, float(current_tpr))

    best_asr = -1.0
    best_threshold = float(scores.min())
    best_tpr = 0.0
    best_fpr = 1.0
    for threshold in np.linspace(float(scores.min()), float(scores.max()), num=2000, endpoint=True):
        preds = (scores >= threshold).astype(np.int64)
        tp = float(((preds == 1) & (labels == 1)).sum())
        tn = float(((preds == 0) & (labels == 0)).sum())
        current_asr = (tp + tn) / float(len(labels))
        if current_asr > best_asr:
            best_asr = current_asr
            best_threshold = float(threshold)
            best_tpr = tp / max(float((labels == 1).sum()), 1.0)
            best_fpr = float(((preds == 1) & (labels == 0)).sum()) / max(float((labels == 0).sum()), 1.0)
    return {
        "auc": auc,
        "asr": float(best_asr),
        "tpr_at_1pct_fpr": float(tpr_at_1pct),
        "threshold": best_threshold,
        "best_tpr": best_tpr,
        "best_fpr": best_fpr,
    }


def analyze_semantic_aux_fusion(records: list[dict[str, object]]) -> dict[str, object]:
    labels = np.asarray([int(row["label"]) for row in records], dtype=int)
    mean_cos = np.asarray([float(row["mean_cos"]) for row in records], dtype=float)
    max_cos = np.asarray([float(row["max_cos"]) for row in records], dtype=float)
    max_ssim = np.asarray([float(row["max_ssim"]) for row in records], dtype=float)

    oriented_mean_cos = _orient_memberness(mean_cos, labels)
    oriented_max_cos = _orient_memberness(max_cos, labels)
    oriented_max_ssim = _orient_memberness(max_ssim, labels)

    candidate_scores = {
        "mean_cos": oriented_mean_cos,
        "max_cos": oriented_max_cos,
        "cosine_pair_zmean": (_zscore(oriented_mean_cos) + _zscore(oriented_max_cos)) / 2.0,
        "cosine_ssim_triplet_zmean": (
            _zscore(oriented_mean_cos) + _zscore(oriented_max_cos) + _zscore(oriented_max_ssim)
        )
        / 3.0,
        "cosine_ssim_rank_fusion": (
            _rankdata(oriented_mean_cos) + _rankdata(oriented_max_cos) + _rankdata(oriented_max_ssim)
        )
        / 3.0,
    }

    candidates: dict[str, dict[str, float]] = {}
    for name, scores in candidate_scores.items():
        metrics = _threshold_metrics(labels, scores)
        metrics["spearman_vs_mean_cos"] = (
            1.0 if name == "mean_cos" else _spearman(candidate_scores["mean_cos"], scores)
        )
        candidates[name] = {key: round(float(value), 6) for key, value in metrics.items()}

    ranked = sorted(
        (
            {
                "name": name,
                "auc": payload["auc"],
                "asr": payload["asr"],
                "spearman_vs_mean_cos": payload["spearman_vs_mean_cos"],
            }
            for name, payload in candidates.items()
        ),
        key=lambda item: (item["auc"], item["asr"]),
        reverse=True,
    )
    best = ranked[0]
    mean_cos_auc = candidates["mean_cos"]["auc"]
    best_auc_gain = round(float(best["auc"] - mean_cos_auc), 6)
    return {
        "sample_count": int(labels.shape[0]),
        "member_rows": int((labels == 1).sum()),
        "nonmember_rows": int((labels == 0).sum()),
        "candidates": candidates,
        "ranked_candidates": ranked,
        "best_candidate": best["name"],
        "best_auc_gain_vs_mean_cos": best_auc_gain,
    }
