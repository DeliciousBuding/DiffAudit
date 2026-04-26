from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def collect_generated_paths(generated_dir: str | Path, sample_index: int, file_name: str) -> list[Path]:
    prefix = f"{sample_index:03d}_{Path(file_name).stem}_ret"
    directory = Path(generated_dir)
    return sorted(path for path in directory.glob(f"{prefix}*.png") if path.is_file())


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


def spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    rank_a = _rankdata(values_a)
    rank_b = _rankdata(values_b)
    std_a = rank_a.std(ddof=0)
    std_b = rank_b.std(ddof=0)
    if std_a == 0.0 or std_b == 0.0:
        return 0.0
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def threshold_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
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


def summarize_prompt_response(labels: np.ndarray, scores: np.ndarray, baseline_scores: np.ndarray) -> dict[str, float]:
    metrics = threshold_metrics(labels, scores)
    metrics["spearman_vs_mean_cos"] = spearman(scores, baseline_scores)
    metrics["auc_gain_vs_mean_cos"] = float(metrics["auc"] - roc_auc_score(labels, baseline_scores))
    return {key: round(float(value), 6) for key, value in metrics.items()}
