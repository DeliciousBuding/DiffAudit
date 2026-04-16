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
    parser.add_argument("--secmi-scores", type=Path, required=True)
    parser.add_argument("--pia-scores", type=Path, default=None)
    parser.add_argument("--control-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--resamples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


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


def build_sample_rows(
    *,
    collections: dict[str, list[int]],
    secmi_payload: dict[str, Any],
    pia_payload: dict[str, Any] | None,
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

    secmi_payload = orient_memberness(load_score_payload(args.secmi_scores))
    pia_payload = orient_memberness(load_score_payload(args.pia_scores)) if args.pia_scores is not None else None

    collections = build_collection_split(
        member_count=len(secmi_payload["member_scores"]),
        nonmember_count=len(secmi_payload["nonmember_scores"]),
        control_size=args.control_size,
        test_size=args.test_size,
    )

    rows = build_sample_rows(
        collections=collections,
        secmi_payload=secmi_payload,
        pia_payload=pia_payload,
    )

    secmi_p_test = _vector(secmi_payload["member_scores"], collections["P_test"])
    secmi_u_test = _vector(secmi_payload["nonmember_scores"], collections["U_test"])
    secmi_stats = one_sided_welch_greater(secmi_p_test, secmi_u_test)

    summary: dict[str, Any] = {
        "status": "completed",
        "track": "gray-box",
        "method": "cdi-internal-canary",
        "surface": "cifar10-ddpm-shared-score-contract",
        "feature_mode": "secmi-stat-only" if pia_payload is None else "paired-pia-secmi",
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
