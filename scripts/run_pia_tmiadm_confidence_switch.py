from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded confidence-gated switching packet on aligned PIA/TMIA-DM scores."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--surface",
        action="append",
        nargs=4,
        metavar=("NAME", "PIA_SCORES", "TMIA_FAMILY_SCORES", "TMIA_FAMILY"),
        help="One aligned surface: name, pia scores.json, tmiadm family-scores.json, tmiadm family key",
    )
    parser.add_argument(
        "--margin-thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    )
    return parser.parse_args()


def load_pia_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(payload["member_scores"], dtype=float), np.asarray(payload["nonmember_scores"], dtype=float)


def load_tmiadm_scores(path: Path, family: str) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    family_payload = payload[family]
    return (
        np.asarray(family_payload["member_scores"], dtype=float),
        np.asarray(family_payload["nonmember_scores"], dtype=float),
    )


def orient_memberness(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    if float(member_scores.mean()) < float(nonmember_scores.mean()):
        return -member_scores, -nonmember_scores, "negated"
    return member_scores, nonmember_scores, "identity"


def zscore(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    if std == 0.0:
        return np.zeros_like(values, dtype=float)
    return (values - float(values.mean())) / std


def roc_curve_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr = roc_curve_points(scores, labels)
    return float(np.trapezoid(tpr, fpr))


def accuracy_best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    sorted_scores = scores[order]
    thresholds = np.r_[sorted_scores[0] - 1e-6, (sorted_scores[:-1] + sorted_scores[1:]) / 2.0, sorted_scores[-1] + 1e-6]
    best = 0.0
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        best = max(best, float((preds == labels).mean()))
    return best


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    fpr, tpr = roc_curve_points(scores, labels)
    valid = tpr[fpr <= target_fpr]
    if valid.size == 0:
        return 0.0
    return float(valid.max())


def build_switched_score(pia_z: np.ndarray, tmiadm_z: np.ndarray, margin_threshold: float) -> np.ndarray:
    zsum = pia_z + tmiadm_z
    margin = np.abs(pia_z - tmiadm_z)
    dominant = np.maximum(pia_z, tmiadm_z)
    return np.where(margin >= margin_threshold, dominant, zsum)


def evaluate_surface(
    pia_member_raw: np.ndarray,
    pia_nonmember_raw: np.ndarray,
    tmiadm_member_raw: np.ndarray,
    tmiadm_nonmember_raw: np.ndarray,
    margin_thresholds: list[float],
) -> dict:
    pia_member, pia_nonmember, pia_orientation = orient_memberness(pia_member_raw, pia_nonmember_raw)
    tmiadm_member, tmiadm_nonmember, tmiadm_orientation = orient_memberness(tmiadm_member_raw, tmiadm_nonmember_raw)

    pia_scores = np.concatenate([pia_member, pia_nonmember])
    tmiadm_scores = np.concatenate([tmiadm_member, tmiadm_nonmember])
    labels = np.concatenate(
        [np.ones(len(pia_member), dtype=int), np.zeros(len(pia_nonmember), dtype=int)]
    )

    pia_z = zscore(pia_scores)
    tmiadm_z = zscore(tmiadm_scores)
    zsum = pia_z + tmiadm_z

    baseline = {
        "pia_auc": round(auc_score(pia_scores, labels), 6),
        "tmiadm_auc": round(auc_score(tmiadm_scores, labels), 6),
        "zsum_auc": round(auc_score(zsum, labels), 6),
        "pia_asr": round(accuracy_best_threshold(pia_scores, labels), 6),
        "tmiadm_asr": round(accuracy_best_threshold(tmiadm_scores, labels), 6),
        "zsum_asr": round(accuracy_best_threshold(zsum, labels), 6),
        "zsum_tpr_at_1fpr": round(tpr_at_fpr(zsum, labels, 0.01), 6),
        "spearman_proxy_corr": round(float(np.corrcoef(np.argsort(np.argsort(pia_scores)), np.argsort(np.argsort(tmiadm_scores)))[0, 1]), 6),
    }

    thresholds = []
    for threshold in margin_thresholds:
        switched = build_switched_score(pia_z, tmiadm_z, threshold)
        thresholds.append(
            {
                "margin_threshold": threshold,
                "auc": round(auc_score(switched, labels), 6),
                "asr": round(accuracy_best_threshold(switched, labels), 6),
                "tpr_at_1fpr": round(tpr_at_fpr(switched, labels, 0.01), 6),
                "dominant_rate": round(float((np.abs(pia_z - tmiadm_z) >= threshold).mean()), 6),
            }
        )

    best = max(thresholds, key=lambda item: (item["auc"], item["asr"]))
    return {
        "orientations": {
            "pia": pia_orientation,
            "tmiadm": tmiadm_orientation,
        },
        "baselines": baseline,
        "threshold_scan": thresholds,
        "best_threshold": best,
    }


def write_analysis(run_root: Path, surfaces: list[dict], selected_threshold: float) -> None:
    lines = [
        "# PIA vs TMIA-DM Confidence-Gated Switching Offline Packet",
        "",
        f"- selected stable threshold: `{selected_threshold}`",
        "",
        "## Surface Summary",
        "",
    ]
    for surface in surfaces:
        best = surface["evaluation"]["best_threshold"]
        base = surface["evaluation"]["baselines"]
        lines.extend(
            [
                f"### `{surface['name']}`",
                f"- `PIA AUC = {base['pia_auc']}`",
                f"- `TMIA-DM AUC = {base['tmiadm_auc']}`",
                f"- `z-score sum AUC = {base['zsum_auc']}`",
                f"- best gated threshold on this surface: `{best['margin_threshold']}`",
                f"- best gated AUC on this surface: `{best['auc']}`",
                "",
            ]
        )
    (run_root / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.surface:
        raise SystemExit("At least one --surface must be provided.")

    args.run_root.mkdir(parents=True, exist_ok=True)
    surfaces = []
    threshold_scores: dict[float, list[float]] = {threshold: [] for threshold in args.margin_thresholds}

    for name, pia_path_str, tmiadm_path_str, family in args.surface:
        pia_path = Path(pia_path_str)
        tmiadm_path = Path(tmiadm_path_str)
        pia_member, pia_nonmember = load_pia_scores(pia_path)
        tmiadm_member, tmiadm_nonmember = load_tmiadm_scores(tmiadm_path, family)
        evaluation = evaluate_surface(
            pia_member_raw=pia_member,
            pia_nonmember_raw=pia_nonmember,
            tmiadm_member_raw=tmiadm_member,
            tmiadm_nonmember_raw=tmiadm_nonmember,
            margin_thresholds=args.margin_thresholds,
        )
        for item in evaluation["threshold_scan"]:
            threshold_scores[item["margin_threshold"]].append(item["auc"])
        surfaces.append(
            {
                "name": name,
                "pia_scores": pia_path.as_posix(),
                "tmiadm_family_scores": tmiadm_path.as_posix(),
                "tmiadm_family": family,
                "evaluation": evaluation,
            }
        )

    selected_threshold = max(
        threshold_scores.items(),
        key=lambda item: (round(mean(item[1]), 6), -abs(item[0] - 0.5)),
    )[0]

    summary = {
        "run_id": args.run_root.name,
        "track": "gray-box",
        "method": "pia-tmiadm-confidence-gated-switching",
        "mode": "offline-packet",
        "status": "completed",
        "policy": {
            "normalized_scores": True,
            "dominant_method_identity": True,
            "margin_thresholds": args.margin_thresholds,
            "selected_threshold": selected_threshold,
            "fallback": "z-score sum",
        },
        "surfaces": surfaces,
        "verdict": (
            "bounded offline packet for confidence-gated switching completed; "
            "selected threshold is based on cross-surface AUC stability, not release-grade generalization."
        ),
        "artifacts": {
            "summary": (args.run_root / "summary.json").as_posix(),
            "analysis": (args.run_root / "analysis.md").as_posix(),
        },
    }
    (args.run_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_analysis(args.run_root, surfaces, selected_threshold)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
