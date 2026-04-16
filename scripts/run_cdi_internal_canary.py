from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ttest_ind


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded internal CDI-style collection canary.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--secmi-scores", type=Path, default=None)
    parser.add_argument("--secmi-run-root", type=Path, default=None)
    parser.add_argument("--pia-scores", type=Path, default=None)
    parser.add_argument(
        "--paired-scorer",
        choices=("auto", "none", "control-z-linear"),
        default="auto",
        help=(
            "Paired scorer policy. 'auto' enables the default internal paired scorer when paired inputs are present; "
            "'none' keeps component-only reporting; 'control-z-linear' forces the current paired scorer explicitly."
        ),
    )
    parser.add_argument("--control-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--resamples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.secmi_scores is None and args.secmi_run_root is None:
        parser.error("One of --secmi-scores or --secmi-run-root is required.")
    return args


def load_score_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    member_scores = [float(v) for v in payload["member_scores"]]
    nonmember_scores = [float(v) for v in payload["nonmember_scores"]]
    member_indices = payload.get("member_indices")
    nonmember_indices = payload.get("nonmember_indices")
    return {
        "member_scores": member_scores,
        "nonmember_scores": nonmember_scores,
        "member_indices": member_indices,
        "nonmember_indices": nonmember_indices,
    }


def load_score_payload_from_run_root(run_root: Path) -> dict[str, Any]:
    member_path = run_root / "secmi_member_scores.npy"
    nonmember_path = run_root / "secmi_nonmember_scores.npy"
    if not member_path.exists() or not nonmember_path.exists():
        raise FileNotFoundError(
            f"Expected SecMI score arrays under {run_root}, but found member={member_path.exists()} "
            f"nonmember={nonmember_path.exists()}."
        )
    return {
        "member_scores": np.load(member_path).astype(float).tolist(),
        "nonmember_scores": np.load(nonmember_path).astype(float).tolist(),
        "member_indices": None,
        "nonmember_indices": None,
    }


def orient_memberness(payload: dict[str, Any]) -> dict[str, Any]:
    member_scores = np.asarray(payload["member_scores"], dtype=float)
    nonmember_scores = np.asarray(payload["nonmember_scores"], dtype=float)
    if float(member_scores.mean()) < float(nonmember_scores.mean()):
        orientation = "negated"
        member_scores = -member_scores
        nonmember_scores = -nonmember_scores
    else:
        orientation = "identity"
    return {
        **payload,
        "member_scores": member_scores.tolist(),
        "nonmember_scores": nonmember_scores.tolist(),
        "memberness_orientation": orientation,
    }


def build_collection_split(
    *,
    member_count: int,
    nonmember_count: int,
    control_size: int,
    test_size: int,
) -> dict[str, list[int]]:
    required = control_size + test_size
    if member_count < required or nonmember_count < required:
        raise ValueError(
            f"Need at least {required} member and non-member scores; got member={member_count}, nonmember={nonmember_count}."
        )
    return {
        "P_ctrl": list(range(0, control_size)),
        "P_test": list(range(control_size, control_size + test_size)),
        "U_ctrl": list(range(0, control_size)),
        "U_test": list(range(control_size, control_size + test_size)),
    }


def _vector(values: list[float], indices: list[int]) -> np.ndarray:
    return np.asarray([float(values[idx]) for idx in indices], dtype=float)


def _safe_std(values: np.ndarray) -> float:
    std = float(np.std(values, ddof=0))
    return std if std > 1e-12 else 1.0


def build_control_z_linear_paired_scorer(
    *,
    collections: dict[str, list[int]],
    secmi_payload: dict[str, Any],
    pia_payload: dict[str, Any],
) -> dict[str, Any]:
    feature_payloads = {
        "secmi_stat": secmi_payload,
        "pia": pia_payload,
    }
    split_map = {
        "P_ctrl": ("member_scores", collections["P_ctrl"]),
        "P_test": ("member_scores", collections["P_test"]),
        "U_ctrl": ("nonmember_scores", collections["U_ctrl"]),
        "U_test": ("nonmember_scores", collections["U_test"]),
    }

    feature_details: dict[str, dict[str, Any]] = {}
    positive_gaps: dict[str, float] = {}
    for feature_name, payload in feature_payloads.items():
        ctrl_member = _vector(payload["member_scores"], collections["P_ctrl"])
        ctrl_nonmember = _vector(payload["nonmember_scores"], collections["U_ctrl"])
        ctrl_all = np.concatenate([ctrl_member, ctrl_nonmember])
        center = float(ctrl_all.mean())
        scale = _safe_std(ctrl_all)

        z_vectors: dict[str, np.ndarray] = {}
        for split_name, (score_key, indices) in split_map.items():
            raw_values = _vector(payload[score_key], indices)
            z_vectors[split_name] = (raw_values - center) / scale

        control_gap = float(z_vectors["P_ctrl"].mean() - z_vectors["U_ctrl"].mean())
        positive_gaps[feature_name] = max(control_gap, 0.0)
        feature_details[feature_name] = {
            "center": center,
            "scale": scale,
            "control_gap": control_gap,
            "z_vectors": z_vectors,
        }

    total_gap = float(sum(positive_gaps.values()))
    if total_gap <= 1e-12:
        weight = 1.0 / float(len(feature_payloads))
        weights = {name: weight for name in feature_payloads}
    else:
        weights = {name: positive_gaps[name] / total_gap for name in feature_payloads}

    paired_vectors: dict[str, np.ndarray] = {}
    for split_name in split_map:
        paired_vectors[split_name] = sum(
            weights[name] * feature_details[name]["z_vectors"][split_name] for name in feature_payloads
        )

    paired_stats = one_sided_welch_greater(paired_vectors["P_test"], paired_vectors["U_test"])
    return {
        "name": "control-z-linear",
        "weights": weights,
        "features": {
            name: {
                "center": details["center"],
                "scale": details["scale"],
                "control_gap": details["control_gap"],
            }
            for name, details in feature_details.items()
        },
        "vectors": paired_vectors,
        "feature_vectors": {
            name: details["z_vectors"] for name, details in feature_details.items()
        },
        "metrics": {
            "paired_p_test_mean": float(paired_vectors["P_test"].mean()),
            "paired_u_test_mean": float(paired_vectors["U_test"].mean()),
            "paired_t_statistic": paired_stats["t_statistic"],
            "paired_p_value": paired_stats["p_value"],
            "paired_control_gap": float(paired_vectors["P_ctrl"].mean() - paired_vectors["U_ctrl"].mean()),
        },
    }


def build_sample_rows(
    *,
    collections: dict[str, list[int]],
    secmi_payload: dict[str, Any],
    pia_payload: dict[str, Any] | None,
    paired_scorer: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_specs = (
        ("P_ctrl", "member", "control"),
        ("P_test", "member", "test"),
        ("U_ctrl", "nonmember", "control"),
        ("U_test", "nonmember", "test"),
    )
    for split_name, source, stage in split_specs:
        score_key = f"{source}_scores"
        index_key = f"{source}_indices"
        payload_indices = secmi_payload.get(index_key)
        for local_idx, source_idx in enumerate(collections[split_name]):
            row: dict[str, Any] = {
                "collection_split": split_name,
                "source_group": source,
                "stage": stage,
                "local_position": local_idx,
                "source_index": int(payload_indices[source_idx]) if payload_indices is not None else int(source_idx),
                "secmi_stat_score": float(secmi_payload[score_key][source_idx]),
            }
            if pia_payload is not None:
                row["pia_score"] = float(pia_payload[score_key][source_idx])
            if paired_scorer is not None:
                row["paired_score"] = float(paired_scorer["vectors"][split_name][local_idx])
                row["secmi_z_score"] = float(paired_scorer["feature_vectors"]["secmi_stat"][split_name][local_idx])
                row["pia_z_score"] = float(paired_scorer["feature_vectors"]["pia"][split_name][local_idx])
            rows.append(row)
    return rows


def one_sided_welch_greater(sample_a: np.ndarray, sample_b: np.ndarray) -> dict[str, float]:
    result = ttest_ind(sample_a, sample_b, equal_var=False)
    statistic = float(result.statistic)
    two_sided = float(result.pvalue)
    if math.isnan(statistic) or math.isnan(two_sided):
        return {"t_statistic": float("nan"), "p_value": float("nan")}
    if statistic > 0:
        p_value = two_sided / 2.0
    else:
        p_value = 1.0 - (two_sided / 2.0)
    return {"t_statistic": statistic, "p_value": float(p_value)}


def run_internal_canary(args: argparse.Namespace) -> dict[str, Any]:
    args.run_root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    raw_secmi_payload = (
        load_score_payload(args.secmi_scores)
        if args.secmi_scores is not None
        else load_score_payload_from_run_root(args.secmi_run_root)
    )
    secmi_payload = orient_memberness(raw_secmi_payload)
    pia_payload = orient_memberness(load_score_payload(args.pia_scores)) if args.pia_scores is not None else None

    collections = build_collection_split(
        member_count=len(secmi_payload["member_scores"]),
        nonmember_count=len(secmi_payload["nonmember_scores"]),
        control_size=args.control_size,
        test_size=args.test_size,
    )

    effective_paired_policy = args.paired_scorer
    paired_scorer = None
    if pia_payload is not None and args.paired_scorer in {"auto", "control-z-linear"}:
        effective_paired_policy = "control-z-linear"
        paired_scorer = build_control_z_linear_paired_scorer(
            collections=collections,
            secmi_payload=secmi_payload,
            pia_payload=pia_payload,
        )
    elif pia_payload is None and args.paired_scorer == "auto":
        effective_paired_policy = "none"

    rows = build_sample_rows(
        collections=collections,
        secmi_payload=secmi_payload,
        pia_payload=pia_payload,
        paired_scorer=paired_scorer,
    )

    secmi_p_test = _vector(secmi_payload["member_scores"], collections["P_test"])
    secmi_u_test = _vector(secmi_payload["nonmember_scores"], collections["U_test"])
    secmi_stats = one_sided_welch_greater(secmi_p_test, secmi_u_test)

    summary: dict[str, Any] = {
        "status": "completed",
        "track": "gray-box",
        "method": "cdi-internal-canary",
        "surface": "cifar10-ddpm-shared-score-contract",
        "feature_mode": (
            "secmi-stat-only"
            if pia_payload is None
            else ("paired-pia-secmi-control-z-linear" if paired_scorer is not None else "paired-pia-secmi")
        ),
        "control_size": args.control_size,
        "test_size": args.test_size,
        "resamples": args.resamples,
        "seed": args.seed,
        "collection_counts": {key: len(value) for key, value in collections.items()},
        "metrics": {
            "secmi_p_test_mean": float(secmi_p_test.mean()),
            "secmi_u_test_mean": float(secmi_u_test.mean()),
            "secmi_t_statistic": secmi_stats["t_statistic"],
            "secmi_p_value": secmi_stats["p_value"],
        },
        "analysis": {
            "secmi_memberness_orientation": secmi_payload["memberness_orientation"],
            "paired_scorer": paired_scorer["name"] if paired_scorer is not None else "none",
            "paired_scorer_policy_requested": args.paired_scorer,
            "paired_scorer_policy_effective": effective_paired_policy,
        },
        "artifacts": {
            "collections": str((args.run_root / "collections.json").as_posix()),
            "sample_scores": str((args.run_root / "sample_scores.jsonl").as_posix()),
            "audit_summary": str((args.run_root / "audit_summary.json").as_posix()),
        },
        "duration_seconds": round(time.perf_counter() - start, 3),
        "notes": [
            "This is an internal CDI-shape canary over existing gray-box score artifacts, not an external copyright claim.",
            "Current first canary keeps the statistic surface as simple as possible before any multi-feature scorer is added.",
        ],
    }

    if pia_payload is not None:
        pia_p_test = _vector(pia_payload["member_scores"], collections["P_test"])
        pia_u_test = _vector(pia_payload["nonmember_scores"], collections["U_test"])
        pia_stats = one_sided_welch_greater(pia_p_test, pia_u_test)
        summary["metrics"].update(
            {
                "pia_p_test_mean": float(pia_p_test.mean()),
                "pia_u_test_mean": float(pia_u_test.mean()),
                "pia_t_statistic": pia_stats["t_statistic"],
                "pia_p_value": pia_stats["p_value"],
            }
        )
        summary["analysis"]["pia_memberness_orientation"] = pia_payload["memberness_orientation"]
    if paired_scorer is not None:
        summary["metrics"].update(paired_scorer["metrics"])
        summary["analysis"]["paired_scorer_details"] = {
            "weights": paired_scorer["weights"],
            "features": paired_scorer["features"],
        }

    (args.run_root / "collections.json").write_text(
        json.dumps({"collections": collections}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (args.run_root / "sample_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.run_root / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    summary = run_internal_canary(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
