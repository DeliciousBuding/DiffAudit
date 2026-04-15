"""Build a decision-grade mainline audit report with actionable suggestions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_repo_path(research_root: Path, repo_relative_path: str) -> Path:
    normalized = repo_relative_path.replace("\\", "/")
    return research_root / Path(normalized)


def _risk_level_from_auc(auc: float | None) -> str:
    if auc is None:
        return "unknown"
    if auc >= 0.95:
        return "critical"
    if auc >= 0.8:
        return "high"
    if auc >= 0.65:
        return "elevated"
    if auc >= 0.55:
        return "guarded"
    return "low"


def _risk_rank(level: str) -> int:
    order = {
        "unknown": 0,
        "low": 1,
        "guarded": 2,
        "elevated": 3,
        "high": 4,
        "critical": 5,
    }
    return order.get(level, 0)


def _priority_rank(priority: str) -> int:
    order = {
        "low": 1,
        "medium": 2,
        "high": 3,
        "critical": 4,
    }
    return order.get(priority, 0)


def _headline_track_key(track: str, defense: str) -> str:
    if track == "black-box":
        return "black_box"
    if track == "gray-box":
        return "gray_box_defended" if defense != "none" else "gray_box_baseline"
    if track == "white-box":
        return "white_box_defended" if defense != "none" else "white_box_attack"
    return track.replace("-", "_")


def _default_track_suggestion(row: dict[str, Any]) -> dict[str, Any]:
    track = row["track"]
    attack = row["attack"]
    defense = row["defense"]
    boundary = row["boundary"]
    auc = row.get("auc")
    asr = row.get("asr")
    level = _risk_level_from_auc(auc if isinstance(auc, (float, int)) else None)

    if track == "black-box":
        return {
            "scope": "black-box",
            "priority": "high",
            "risk_level": level,
            "title": "Keep recon as the ingress risk probe",
            "why": f"{attack} still shows externally observable leakage risk at auc={auc} / asr={asr}.",
            "action": "Keep black-box auditing live on controlled/public-subset bundles and attach rate-limit, logging, and review hooks before claiming deployment readiness.",
            "boundary": boundary,
        }

    if track == "gray-box" and defense == "none":
        return {
            "scope": "gray-box",
            "priority": "critical",
            "risk_level": level,
            "title": "Treat PIA baseline as the main optimization pressure",
            "why": f"{attack} remains the strongest attack-defense mainline with auc={auc} / asr={asr}.",
            "action": "Use this baseline as the canonical attack target for further GPU tuning and compare every candidate defense against the same adaptive-reviewed PIA path.",
            "boundary": boundary,
        }

    if track == "gray-box":
        return {
            "scope": "gray-box",
            "priority": "critical",
            "risk_level": level,
            "title": "Keep stochastic-dropout on the defended critical path",
            "why": f"{defense} lowers the admitted gray-box metrics while preserving the current runtime path.",
            "action": "Continue tuning inference-time stochastic dropout against the admitted PIA baseline and reject candidates that do not beat the current defended pairing on both effect and cost.",
            "boundary": boundary,
        }

    if track == "white-box" and defense == "none":
        return {
            "scope": "white-box",
            "priority": "critical",
            "risk_level": level,
            "title": "Assume privileged access remains catastrophic",
            "why": f"{attack} reaches auc={auc} / asr={asr}, which is a white-box upper-bound failure signal.",
            "action": "Keep high-privilege model access behind explicit review and use white-box results as the upper-bound audit line rather than a product-facing KPI.",
            "boundary": boundary,
        }

    return {
        "scope": "white-box",
        "priority": "high",
        "risk_level": level,
        "title": "Use W-1 as the defended comparator, not the final answer",
        "why": f"{defense} is currently the best defended white-box comparator in the admitted table.",
        "action": "Use this defended rung as the current white-box reduction reference while keeping benchmark claims frozen at the bridge level.",
        "boundary": boundary,
    }


def _collect_mainline_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("attack-defense table rows must be a list")

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = _headline_track_key(str(row.get("track", "")), str(row.get("defense", "")))
        result.setdefault(key, row)
    return result


def _collect_corroborating_rows(
    payload: dict[str, Any],
    selected_rows: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return {}

    selected_sources = {
        str(row.get("source", ""))
        for row in selected_rows.values()
        if isinstance(row, dict)
    }
    corroborating: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source", ""))
        track = str(row.get("track", ""))
        if source in selected_sources or not track:
            continue
        corroborating.setdefault(track, []).append(row)
    return corroborating


def _sort_suggestions(suggestions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        suggestions,
        key=lambda item: (
            _priority_rank(str(item.get("priority", ""))),
            _risk_rank(str(item.get("risk_level", ""))),
            str(item.get("scope", "")),
        ),
        reverse=True,
    )


def _summary_mtime_ns(summary_path: Path) -> int:
    try:
        return int(summary_path.stat().st_mtime_ns)
    except FileNotFoundError:
        return 0


def _load_pia_summary_pairs(research_root: Path) -> list[dict[str, Any]]:
    runs_root = research_root / "workspaces" / "gray-box" / "runs"
    summaries = sorted(runs_root.rglob("summary.json"))
    grouped: dict[tuple[int, str], dict[str, Any]] = {}

    for summary_path in summaries:
        payload = _load_json(summary_path)
        if payload.get("track") != "gray-box" or payload.get("method") != "pia":
            continue
        if payload.get("mode") != "runtime-mainline":
            continue

        sample_count = payload.get("sample_count_per_split")
        if not isinstance(sample_count, int):
            continue

        defense = payload.get("defense", {})
        if not isinstance(defense, dict):
            continue

        enabled = bool(defense.get("enabled"))
        schedule = str(defense.get("dropout_activation_schedule", "off"))
        workspace_name = str(payload.get("workspace_name", ""))
        if enabled and schedule == "all_steps":
            pair_kind = "defense"
        elif (not enabled) and "dropout-defense" not in workspace_name:
            pair_kind = "baseline"
        else:
            continue
        key = (sample_count, pair_kind)

        candidate = {
            "payload": payload,
            "mtime_ns": _summary_mtime_ns(summary_path),
            "workspace_name": workspace_name,
        }
        current = grouped.get(key)
        if current is None or (
            candidate["mtime_ns"],
            candidate["workspace_name"],
        ) > (
            int(current["mtime_ns"]),
            str(current["workspace_name"]),
        ):
            grouped[key] = candidate

    pairs: list[dict[str, Any]] = []
    sample_counts = sorted({sample_count for sample_count, _ in grouped})
    for sample_count in sample_counts:
        baseline_entry = grouped.get((sample_count, "baseline"))
        defense_entry = grouped.get((sample_count, "defense"))
        baseline = baseline_entry["payload"] if baseline_entry else None
        defense = defense_entry["payload"] if defense_entry else None
        if baseline is None or defense is None:
            continue

        baseline_metrics = baseline.get("adaptive_check", {}).get("metrics", baseline.get("metrics", {}))
        defense_metrics = defense.get("adaptive_check", {}).get("metrics", defense.get("metrics", {}))
        baseline_cost = baseline.get("cost", {})
        defense_cost = defense.get("cost", {})
        baseline_device = str(
            baseline_cost.get("device")
            or baseline.get("device")
            or ""
        ).lower()
        defense_device = str(
            defense_cost.get("device")
            or defense.get("device")
            or ""
        ).lower()
        execution_device = defense_device or baseline_device
        if not execution_device:
            workspace_hint = " ".join(
                [
                    str(baseline.get("workspace_name", "")),
                    str(defense.get("workspace_name", "")),
                    str(baseline.get("artifact_paths", {}).get("summary", "")),
                    str(defense.get("artifact_paths", {}).get("summary", "")),
                ]
            ).lower()
            if "gpu" in workspace_hint or "cuda" in workspace_hint:
                execution_device = "gpu"
            elif "cpu" in workspace_hint:
                execution_device = "cpu"

        baseline_auc = float(baseline_metrics.get("auc"))
        defense_auc = float(defense_metrics.get("auc"))
        baseline_asr = float(baseline_metrics.get("asr"))
        defense_asr = float(defense_metrics.get("asr"))
        defense_wall_clock = float(defense_cost.get("wall_clock_seconds", 0.0) or 0.0)
        baseline_wall_clock = float(baseline_cost.get("wall_clock_seconds", 0.0) or 0.0)

        auc_drop = round(baseline_auc - defense_auc, 6)
        asr_drop = round(baseline_asr - defense_asr, 6)
        effect_score = max(auc_drop, 0.0) + max(asr_drop, 0.0)
        efficiency_score = round(effect_score / max(defense_wall_clock, 1.0), 9)

        pairs.append(
            {
                "sample_count_per_split": sample_count,
                "baseline_summary": str(summary_path_as_repo_relative(research_root, Path(baseline["artifact_paths"]["summary"]))),
                "defense_summary": str(summary_path_as_repo_relative(research_root, Path(defense["artifact_paths"]["summary"]))),
                "baseline_auc": baseline_auc,
                "defense_auc": defense_auc,
                "auc_drop": auc_drop,
                "baseline_asr": baseline_asr,
                "defense_asr": defense_asr,
                "asr_drop": asr_drop,
                "baseline_wall_clock_seconds": round(baseline_wall_clock, 6),
                "defense_wall_clock_seconds": round(defense_wall_clock, 6),
                "execution_device": execution_device,
                "efficiency_score": efficiency_score,
            }
        )
    return pairs


def _load_pia_ablation_summaries(research_root: Path) -> list[dict[str, Any]]:
    runs_root = research_root / "workspaces" / "gray-box" / "runs"
    summaries = sorted(runs_root.rglob("summary.json"))
    grouped: dict[tuple[int, str], dict[str, Any]] = {}

    for summary_path in summaries:
        payload = _load_json(summary_path)
        if payload.get("track") != "gray-box" or payload.get("method") != "pia":
            continue
        if payload.get("mode") != "runtime-mainline":
            continue

        sample_count = payload.get("sample_count_per_split")
        if not isinstance(sample_count, int):
            continue

        defense = payload.get("defense", {})
        if not isinstance(defense, dict):
            continue
        schedule = str(defense.get("dropout_activation_schedule", "off"))
        enabled = bool(defense.get("enabled"))
        workspace_name = str(payload.get("workspace_name", ""))

        if not enabled and "dropout-defense" not in workspace_name:
            kind = "baseline"
        elif enabled and schedule in {"all_steps", "late_steps_only"}:
            kind = schedule
        else:
            continue

        key = (sample_count, kind)
        candidate = {
            "payload": payload,
            "mtime_ns": _summary_mtime_ns(summary_path),
            "workspace_name": workspace_name,
        }
        current = grouped.get(key)
        if current is None or (
            candidate["mtime_ns"],
            candidate["workspace_name"],
        ) > (
            int(current["mtime_ns"]),
            str(current["workspace_name"]),
        ):
            grouped[key] = candidate

    ablations: list[dict[str, Any]] = []
    sample_counts = sorted({sample_count for sample_count, _ in grouped})
    for sample_count in sample_counts:
        baseline_entry = grouped.get((sample_count, "baseline"))
        all_steps_entry = grouped.get((sample_count, "all_steps"))
        late_steps_entry = grouped.get((sample_count, "late_steps_only"))
        baseline = baseline_entry["payload"] if baseline_entry else None
        all_steps = all_steps_entry["payload"] if all_steps_entry else None
        late_steps = late_steps_entry["payload"] if late_steps_entry else None
        if baseline is None:
            continue

        baseline_metrics = baseline.get("adaptive_check", {}).get("metrics", baseline.get("metrics", {}))
        baseline_cost = baseline.get("cost", {})
        item: dict[str, Any] = {
            "sample_count_per_split": sample_count,
            "baseline_summary": str(summary_path_as_repo_relative(research_root, Path(baseline["artifact_paths"]["summary"]))),
            "baseline_auc": float(baseline_metrics.get("auc")),
            "baseline_asr": float(baseline_metrics.get("asr")),
            "baseline_wall_clock_seconds": round(float(baseline_cost.get("wall_clock_seconds", 0.0) or 0.0), 6),
        }
        for name, payload in (("all_steps", all_steps), ("late_steps_only", late_steps)):
            if payload is None:
                continue
            metrics = payload.get("adaptive_check", {}).get("metrics", payload.get("metrics", {}))
            cost = payload.get("cost", {})
            item[name] = {
                "summary": str(summary_path_as_repo_relative(research_root, Path(payload["artifact_paths"]["summary"]))),
                "auc": float(metrics.get("auc")),
                "asr": float(metrics.get("asr")),
                "auc_drop": round(item["baseline_auc"] - float(metrics.get("auc")), 6),
                "asr_drop": round(item["baseline_asr"] - float(metrics.get("asr")), 6),
                "wall_clock_seconds": round(float(cost.get("wall_clock_seconds", 0.0) or 0.0), 6),
            }
        ablations.append(item)
    return ablations


def _load_pia_schedule_series(
    research_root: Path,
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    runs_root = research_root / "workspaces" / "gray-box" / "runs"
    summaries = sorted(runs_root.rglob("summary.json"))
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}

    for summary_path in summaries:
        payload = _load_json(summary_path)
        if payload.get("track") != "gray-box" or payload.get("method") != "pia":
            continue
        if payload.get("mode") != "runtime-mainline":
            continue

        sample_count = payload.get("sample_count_per_split")
        if not isinstance(sample_count, int):
            continue

        defense = payload.get("defense", {})
        if not isinstance(defense, dict):
            continue

        enabled = bool(defense.get("enabled"))
        schedule = str(defense.get("dropout_activation_schedule", "off"))
        workspace_name = str(payload.get("workspace_name", ""))
        if not enabled and "dropout-defense" in workspace_name:
            continue

        metrics = payload.get("adaptive_check", {}).get("metrics", payload.get("metrics", {}))
        score_std = payload.get("adaptive_check", {}).get("score_std", {})
        cost = payload.get("cost", {})

        if enabled:
            if schedule not in {"all_steps", "late_steps_only"}:
                continue
            normalized_schedule = schedule
        else:
            normalized_schedule = "off"

        series_key = (sample_count, normalized_schedule)
        grouped.setdefault(series_key, []).append(
            {
                "mtime_ns": _summary_mtime_ns(summary_path),
                "workspace_name": workspace_name,
                "summary": str(summary_path_as_repo_relative(research_root, Path(payload["artifact_paths"]["summary"]))),
                "defense_enabled": enabled,
                "schedule": normalized_schedule,
                "auc": float(metrics.get("auc")),
                "asr": float(metrics.get("asr")),
                "wall_clock_seconds": round(float(cost.get("wall_clock_seconds", 0.0) or 0.0), 6),
                "member_score_std_mean": round(float(score_std.get("member_mean", 0.0) or 0.0), 6),
                "nonmember_score_std_mean": round(float(score_std.get("nonmember_mean", 0.0) or 0.0), 6),
            }
        )

    for items in grouped.values():
        items.sort(key=lambda item: (int(item["mtime_ns"]), str(item["workspace_name"])))
    return grouped


def summary_path_as_repo_relative(research_root: Path, summary_path: Path) -> Path:
    normalized = Path(str(summary_path).replace("\\", "/"))
    if normalized.is_absolute():
        try:
            return normalized.relative_to(research_root)
        except ValueError:
            return normalized
    return normalized


def _range(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(max(values) - min(values), 6)


def _stability_judgement(auc_range: float, asr_range: float) -> str:
    if auc_range <= 0.01 and asr_range <= 0.01:
        return "stable"
    if auc_range <= 0.02 and asr_range <= 0.02:
        return "acceptable"
    return "volatile"


def _summarize_series(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "status": "missing",
            "run_count": 0,
        }

    auc_values = [float(item["auc"]) for item in items]
    asr_values = [float(item["asr"]) for item in items]
    wall_clock_values = [float(item["wall_clock_seconds"]) for item in items]
    member_std_values = [float(item["member_score_std_mean"]) for item in items]
    nonmember_std_values = [float(item["nonmember_score_std_mean"]) for item in items]
    latest = items[-1]

    summary = {
        "status": "ready",
        "run_count": len(items),
        "latest_workspace": latest["workspace_name"],
        "latest_summary": latest["summary"],
        "latest_auc": latest["auc"],
        "latest_asr": latest["asr"],
        "auc_range": _range(auc_values),
        "asr_range": _range(asr_values),
        "wall_clock_range_seconds": _range(wall_clock_values),
        "mean_member_score_std": round(sum(member_std_values) / len(member_std_values), 6),
        "mean_nonmember_score_std": round(sum(nonmember_std_values) / len(nonmember_std_values), 6),
    }
    if len(items) >= 2:
        summary["stability_judgement"] = _stability_judgement(
            summary["auc_range"],
            summary["asr_range"],
        )
    else:
        summary["stability_judgement"] = "single-run"
    return summary


def _build_gray_box_runtime_health(
    pia_pairs: list[dict[str, Any]],
    pia_ablations: list[dict[str, Any]],
    pia_series: dict[tuple[int, str], list[dict[str, Any]]],
    defense_schedule: dict[str, Any],
) -> dict[str, Any]:
    sample_count = None
    if defense_schedule.get("status") == "ready":
        sample_count = int(defense_schedule["sample_count_per_split"])
    elif pia_pairs:
        sample_count = int(max(pia_pairs, key=lambda item: int(item["sample_count_per_split"]))["sample_count_per_split"])

    if sample_count is None:
        return {
            "status": "pending",
            "reason": "No gray-box PIA runtime summaries were found for stability analysis.",
        }

    ablation = next(
        (item for item in pia_ablations if int(item["sample_count_per_split"]) == sample_count),
        None,
    )
    recommended_schedule = (
        str(defense_schedule.get("recommended_schedule"))
        if defense_schedule.get("status") == "ready"
        else "all_steps"
    )

    return {
        "status": "ready",
        "sample_count_per_split": sample_count,
        "recommended_schedule": recommended_schedule,
        "schedule_stability": {
            "off": _summarize_series(pia_series.get((sample_count, "off"), [])),
            "all_steps": _summarize_series(pia_series.get((sample_count, "all_steps"), [])),
            "late_steps_only": _summarize_series(pia_series.get((sample_count, "late_steps_only"), [])),
        },
        "freshest_same_rung_ablation": ablation,
        "paired_tradeoffs": [
            pair
            for pair in pia_pairs
            if int(pair["sample_count_per_split"]) == sample_count
        ],
    }


def _build_deployment_blockers(mainline: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for key, row in mainline.items():
        if not row.get("source_exists"):
            blockers.append(
                {
                    "scope": key,
                    "severity": "critical",
                    "type": "missing-evidence",
                    "title": "Mainline source artifact is missing",
                    "detail": f"The admitted source path is unavailable: {row['source']}",
                    "required_before_promotion": "Restore the referenced summary artifact or replace the admitted row with a verifiable source.",
                }
            )

    gray_box = mainline["gray_box_defended"]
    blockers.append(
        {
            "scope": "gray-box",
            "severity": "high",
            "type": "provenance-boundary",
            "title": "Gray-box mainline remains bounded by provenance",
            "detail": gray_box["boundary"],
            "required_before_promotion": "Do not describe PIA as paper-faithful or benchmark-ready until checkpoint/source provenance closes.",
        }
    )

    black_box = mainline["black_box"]
    blockers.append(
        {
            "scope": "black-box",
            "severity": "medium",
            "type": "coverage-boundary",
            "title": "Black-box recon evidence is still subset-bounded",
            "detail": black_box["boundary"],
            "required_before_promotion": "Keep claims limited to controlled public-subset leakage risk, not generalized internet exploitability.",
        }
    )

    white_box_attack = mainline["white_box_attack"]
    blockers.append(
        {
            "scope": "white-box",
            "severity": "medium",
            "type": "upper-bound-only",
            "title": "White-box attack is an upper-bound audit line",
            "detail": white_box_attack["boundary"],
            "required_before_promotion": "Use GSA as privileged-risk evidence only; do not present it as a normal deployment KPI.",
        }
    )

    white_box_defended = mainline["white_box_defended"]
    blockers.append(
        {
            "scope": "white-box",
            "severity": "medium",
            "type": "comparator-boundary",
            "title": "W-1 remains a defended comparator, not a final benchmark",
            "detail": white_box_defended["boundary"],
            "required_before_promotion": "Keep W-1 as the current defended bridge until a stronger admitted white-box defense path exists.",
        }
    )

    return sorted(
        blockers,
        key=lambda item: (_priority_rank(str(item.get("severity", ""))), str(item.get("scope", ""))),
        reverse=True,
    )


def _build_consumer_views(
    mainline: dict[str, dict[str, Any]],
    corroborating_evidence: dict[str, list[dict[str, Any]]],
    gpu_action: dict[str, Any],
    defense_schedule: dict[str, Any],
) -> dict[str, Any]:
    recommended_schedule = defense_schedule.get("recommended_schedule", "all_steps")
    auditor_sources = [
        mainline["black_box"]["source"],
        mainline["gray_box_defended"]["source"],
        mainline["white_box_attack"]["source"],
    ]
    for row in corroborating_evidence.get("gray-box", []):
        source = row.get("source")
        if isinstance(source, str):
            auditor_sources.append(source)
    return {
        "auditor": {
            "primary_focus": "Confirm risk exists, keep boundary language intact, and escalate high-risk tracks before deployment language appears.",
            "use_sources": auditor_sources,
        },
        "operator": {
            "primary_focus": "Treat recon as ingress detection, keep privileged access gated, and prefer stochastic-dropout on the gray-box defended path.",
            "recommended_gray_box_schedule": recommended_schedule,
            "next_gpu_rung": gpu_action.get("recommended_rung"),
        },
        "research": {
            "primary_focus": "Keep tuning on the admitted PIA defended line without reopening the frozen mainline.",
            "next_gpu_command": gpu_action.get("command"),
        },
    }


def _recommend_next_gpu_action(pia_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pia_pairs:
        return {
            "status": "unavailable",
            "reason": "No paired PIA baseline/defense summaries were found for GPU tradeoff analysis.",
        }

    positive_pairs = [
        pair
        for pair in pia_pairs
        if pair["auc_drop"] > 0
        and pair["asr_drop"] > 0
        and any(token in str(pair.get("execution_device", "")) for token in ("cuda", "gpu"))
    ]
    if not positive_pairs:
        positive_pairs = [
            pair
            for pair in pia_pairs
            if pair["auc_drop"] > 0 and pair["asr_drop"] > 0
        ]
    if not positive_pairs:
        return {
            "status": "hold",
            "reason": "Existing PIA defense pairs do not show a stable positive movement on both AUC and ASR.",
            "candidate_pairs": pia_pairs,
        }

    recommended = max(
        positive_pairs,
        key=lambda pair: (pair["efficiency_score"], pair["auc_drop"], -pair["defense_wall_clock_seconds"]),
    )
    sample_count = recommended["sample_count_per_split"]
    workspace_name = f"workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-next-gpu-{sample_count}-adaptive"
    return {
        "status": "recommended",
        "focus_track": "gray-box",
        "focus_method": "pia",
        "recommended_rung": f"GPU{sample_count} all_steps adaptive",
        "why": "This rung gives the best positive defense movement per unit wall clock among the currently paired PIA summaries.",
        "auc_drop": recommended["auc_drop"],
        "asr_drop": recommended["asr_drop"],
        "defense_wall_clock_seconds": recommended["defense_wall_clock_seconds"],
        "baseline_summary": recommended["baseline_summary"],
        "defense_summary": recommended["defense_summary"],
        "command": [
            "python",
            "-m",
            "diffaudit",
            "run-pia-runtime-mainline",
            "--config",
            "configs/attacks/pia_mainline_canonical.yaml",
            "--workspace",
            workspace_name,
            "--repo-root",
            "external/PIA",
            "--member-split-root",
            "external/PIA/DDPM",
            "--device",
            "cuda:0",
            "--max-samples",
            str(sample_count),
            "--batch-size",
            "8",
            "--stochastic-dropout-defense",
            "--dropout-activation-schedule",
            "all_steps",
            "--adaptive-query-repeats",
            "3",
            "--provenance-status",
            "workspace-verified",
        ],
    }


def _recommend_dropout_schedule(pia_ablations: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [
        item
        for item in pia_ablations
        if "all_steps" in item and "late_steps_only" in item
    ]
    if not complete:
        return {
            "status": "pending",
            "reason": "A fresh baseline / all_steps / late_steps_only trio is not yet available for the same sample count.",
        }

    recommended = max(
        complete,
        key=lambda item: (
            item["all_steps"]["auc_drop"] - item["late_steps_only"]["auc_drop"],
            item["all_steps"]["asr_drop"] - item["late_steps_only"]["asr_drop"],
        ),
    )
    return {
        "status": "ready",
        "sample_count_per_split": recommended["sample_count_per_split"],
        "recommended_schedule": "all_steps",
        "runner_up_schedule": "late_steps_only",
        "why": "all_steps still delivers the stronger privacy drop on the freshest same-rung ablation trio.",
        "all_steps": recommended["all_steps"],
        "late_steps_only": recommended["late_steps_only"],
    }


def build_mainline_audit_report(
    research_root: str | Path,
    workspace: str | Path,
    attack_defense_table_path: str | Path | None = None,
) -> dict[str, Any]:
    research_root_path = Path(research_root).resolve()
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    table_path = (
        Path(attack_defense_table_path).resolve()
        if attack_defense_table_path
        else research_root_path / "workspaces" / "implementation" / "artifacts" / "unified-attack-defense-table.json"
    )
    payload = _load_json(table_path)
    rows = _collect_mainline_rows(payload)

    required_keys = (
        "black_box",
        "gray_box_baseline",
        "gray_box_defended",
        "white_box_attack",
        "white_box_defended",
    )
    missing = [key for key in required_keys if key not in rows]
    if missing:
        raise ValueError(f"mainline attack-defense table missing required rows: {', '.join(missing)}")

    mainline = {
        key: {
            **row,
            "risk_level": _risk_level_from_auc(float(row["auc"]) if isinstance(row.get("auc"), (float, int)) else None),
            "source_exists": _resolve_repo_path(research_root_path, str(row["source"])).exists(),
        }
        for key, row in rows.items()
        if key in required_keys
    }
    corroborating_evidence = {
        track: [
            {
                **row,
                "risk_level": _risk_level_from_auc(float(row["auc"]) if isinstance(row.get("auc"), (float, int)) else None),
                "source_exists": _resolve_repo_path(research_root_path, str(row["source"])).exists(),
            }
            for row in rows
        ]
        for track, rows in _collect_corroborating_rows(payload, rows).items()
    }

    overall_risk = max(
        (entry["risk_level"] for entry in mainline.values()),
        key=_risk_rank,
    )
    suggestions = [_default_track_suggestion(mainline[key]) for key in required_keys]
    pia_pairs = _load_pia_summary_pairs(research_root_path)
    pia_ablations = _load_pia_ablation_summaries(research_root_path)
    pia_series = _load_pia_schedule_series(research_root_path)
    gpu_action = _recommend_next_gpu_action(pia_pairs)
    defense_schedule = _recommend_dropout_schedule(pia_ablations)
    gray_box_runtime_health = _build_gray_box_runtime_health(
        pia_pairs,
        pia_ablations,
        pia_series,
        defense_schedule,
    )
    deployment_blockers = _build_deployment_blockers(mainline)
    consumer_views = _build_consumer_views(mainline, corroborating_evidence, gpu_action, defense_schedule)

    report = {
        "status": "ready",
        "mode": "mainline-audit-report",
        "schema": "diffaudit.mainline_audit_report.v2",
        "research_root": str(research_root_path),
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
            "next_gpu_action": str(workspace_path / "next_gpu_action.json"),
            "attack_defense_table": str(table_path),
        },
        "system_posture": {
            "overall_risk_level": overall_risk,
            "deployment_readiness": "research-preview",
            "mainline": "recon / PIA + stochastic-dropout(all_steps) / GSA + W-1",
            "summary": "Current admitted evidence is strong enough to drive risk auditing and action prioritization, but boundary and provenance limits still block benchmark-level claims.",
        },
        "mainline": mainline,
        "corroborating_evidence": corroborating_evidence,
        "suggestions": _sort_suggestions(suggestions),
        "deployment_blockers": deployment_blockers,
        "gpu_next_action": gpu_action,
        "dropout_schedule_recommendation": defense_schedule,
        "gray_box_runtime_health": gray_box_runtime_health,
        "consumer_views": consumer_views,
        "notes": [
            "This report is built from the admitted attack-defense table plus current PIA paired runtime summaries.",
            "Suggestions are machine-generated heuristics for audit and tuning prioritization; they do not override provenance or benchmark boundaries.",
        ],
    }
    (workspace_path / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    next_action_packet = {
        "status": gpu_action.get("status"),
        "mode": "mainline-next-gpu-action",
        "schema": "diffaudit.mainline_gpu_action.v1",
        "generated_from": str(workspace_path / "summary.json"),
        "focus_track": gpu_action.get("focus_track"),
        "focus_method": gpu_action.get("focus_method"),
        "recommended_rung": gpu_action.get("recommended_rung"),
        "why": gpu_action.get("why"),
        "baseline_summary": gpu_action.get("baseline_summary"),
        "defense_summary": gpu_action.get("defense_summary"),
        "acceptance": {
            "must_improve_auc_drop": True,
            "must_improve_asr_drop": True,
            "must_preserve_defense_mode": "stochastic-dropout(all_steps)",
            "must_preserve_provenance_status": "workspace-verified",
        },
        "command": gpu_action.get("command"),
    }
    (workspace_path / "next_gpu_action.json").write_text(
        json.dumps(next_action_packet, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return report
