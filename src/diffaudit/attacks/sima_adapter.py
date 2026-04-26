"""Bounded SimA runtime feasibility on the current DDPM/CIFAR-10 asset line."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diffaudit.attacks.pia_adapter import (
    PiaDefaults,
    _build_pia_packet_loader,
    _build_pia_subset_loader,
    _compute_auc,
    _compute_threshold_metrics,
    _load_packet_indices_file,
    _validate_packet_indices,
    build_pia_plan,
    load_pia_model,
    load_pia_ddpm_modules,
    prepare_pia_runtime,
    probe_pia_assets,
    validate_pia_workspace,
)
from diffaudit.config import AuditConfig


def _compute_sima_scores_from_eps(
    eps_prediction: torch.Tensor,
    p_norm: int | float = 4,
) -> torch.Tensor:
    flattened = eps_prediction.flatten(start_dim=1)
    norms = torch.linalg.vector_norm(flattened, ord=p_norm, dim=1)
    return (-norms).detach().cpu()


def _alpha_bar_schedule(
    defaults: PiaDefaults | None = None,
) -> torch.Tensor:
    config = defaults or PiaDefaults()
    betas = torch.linspace(config.beta_1, config.beta_T, config.T, dtype=torch.float32)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def _score_loader_for_timestep(
    model: torch.nn.Module,
    loader,
    timestep: int,
    alpha_bars: torch.Tensor,
    device: str,
    p_norm: int | float,
    noise_seed: int,
) -> torch.Tensor:
    scores: list[torch.Tensor] = []
    alpha_bar_t = alpha_bars[int(timestep)].to(device=device)
    sqrt_alpha = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_bar_t)
    generator = torch.Generator(device=device if device.startswith("cuda") else "cpu")
    generator.manual_seed(int(noise_seed) + int(timestep))
    model.eval()
    with torch.no_grad():
        for batch, _ in loader:
            normalized = batch.to(device) * 2 - 1
            noise = torch.randn(
                normalized.shape,
                generator=generator,
                device=device,
                dtype=normalized.dtype,
            )
            noisy = sqrt_alpha * normalized + sqrt_one_minus_alpha * noise
            timestep_tensor = torch.full(
                (normalized.shape[0],),
                fill_value=int(timestep),
                device=device,
                dtype=torch.long,
            )
            eps_prediction = model(noisy, t=timestep_tensor)
            scores.append(_compute_sima_scores_from_eps(eps_prediction, p_norm=p_norm))
    return torch.cat(scores, dim=0)


def run_sima_runtime_feasibility(
    config: AuditConfig,
    workspace: str | Path,
    repo_root: str | Path = "external/PIA",
    member_split_root: str | Path | None = None,
    device: str = "cpu",
    max_samples: int | None = None,
    batch_size: int = 8,
    scan_timesteps: list[int] | tuple[int, ...] | None = None,
    p_norm: int | float = 4,
    noise_seed: int = 0,
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

    scan_points = [int(t) for t in (scan_timesteps or [20, 40, 60, 80, 100, 120])]
    alpha_bars = _alpha_bar_schedule()
    scan_results: list[dict[str, Any]] = []
    best_entry: dict[str, Any] | None = None
    for timestep in scan_points:
        member_scores = _score_loader_for_timestep(
            model,
            member_loader,
            timestep=timestep,
            alpha_bars=alpha_bars,
            device=device,
            p_norm=p_norm,
            noise_seed=noise_seed,
        )
        nonmember_scores = _score_loader_for_timestep(
            model,
            nonmember_loader,
            timestep=timestep,
            alpha_bars=alpha_bars,
            device=device,
            p_norm=p_norm,
            noise_seed=noise_seed,
        )
        metrics = {
            "auc": round(_compute_auc(member_scores, nonmember_scores), 6),
            **_compute_threshold_metrics(
                member_scores.numpy(),
                nonmember_scores.numpy(),
            ),
            "member_score_mean": round(float(member_scores.mean().item()), 6),
            "nonmember_score_mean": round(float(nonmember_scores.mean().item()), 6),
        }
        entry = {
            "timestep": int(timestep),
            "metrics": metrics,
        }
        scan_results.append(entry)
        if best_entry is None:
            best_entry = entry
        else:
            if (
                entry["metrics"]["auc"] > best_entry["metrics"]["auc"]
                or (
                    entry["metrics"]["auc"] == best_entry["metrics"]["auc"]
                    and entry["timestep"] < best_entry["timestep"]
                )
            ):
                best_entry = entry

    if best_entry is None:
        raise RuntimeError("SimA feasibility scan produced no results")

    scores_path = workspace_path / "scan-results.json"
    scores_path.write_text(json.dumps(scan_results, indent=2, ensure_ascii=True), encoding="utf-8")

    result = {
        "status": "ready",
        "track": "gray-box",
        "method": "sima",
        "paper": "SimA_arXiv2025",
        "mode": "runtime-feasibility",
        "device": device,
        "workspace": str(workspace_path),
        "workspace_name": workspace_path.name,
        "contract_stage": "target",
        "asset_grade": "single-machine-real-asset",
        "provenance_status": provenance_status,
        "evidence_level": "feasibility",
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
            "scan_results": str(scores_path),
        },
        "checks": {
            **asset_summary["checks"],
            "runtime_probe_ready": True,
            "scan_completed": True,
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
            "scan_timesteps": scan_points,
            "p_norm": float(p_norm),
            "noise_seed": int(noise_seed),
            "elapsed_seconds": round(time.perf_counter() - started_at, 6),
        },
        "sample_count_per_split": int(len(member_indices)),
        "best_timestep": int(best_entry["timestep"]),
        "metrics": best_entry["metrics"],
        "scan_results": scan_results,
        "notes": [
            "SimA feasibility uses a single-query score-norm statistic over DDPM denoiser outputs.",
            "This is a bounded local feasibility result, not yet a promoted gray-box challenger verdict.",
        ],
    }
    (workspace_path / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return result


def export_sima_packet_scores(
    config: AuditConfig,
    workspace: str | Path,
    repo_root: str | Path = "external/PIA",
    member_split_root: str | Path | None = None,
    device: str = "cpu",
    packet_size: int = 4,
    member_offset: int = 0,
    nonmember_offset: int = 0,
    member_index_file: str | Path | None = None,
    nonmember_index_file: str | Path | None = None,
    batch_size: int = 4,
    timestep: int = 20,
    p_norm: int | float = 4,
    noise_seed: int = 0,
    provenance_status: str = "workspace-verified",
) -> dict[str, Any]:
    if device.lower() != "cpu":
        raise ValueError("SimA packet export is CPU-first and does not authorize GPU use.")

    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    plan = build_pia_plan(config)
    validate_pia_workspace(repo_root)
    split_root = Path(member_split_root) if member_split_root else Path(repo_root) / "DDPM"
    asset_summary = probe_pia_assets(
        dataset=plan.dataset,
        dataset_root=plan.data_root,
        model_dir=plan.model_dir,
        member_split_root=split_root,
    )
    if asset_summary["status"] != "ready":
        result = {
            "status": "blocked",
            "track": "gray-box",
            "method": "sima",
            "paper": "SimA_arXiv2025",
            "mode": "packet-score-export",
            "gpu_release": "none",
            "admitted_change": "none",
            "asset_probe": asset_summary,
            "artifact_paths": {
                "summary": str(workspace_path / "summary.json"),
            },
        }
        (workspace_path / "summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return result

    _, model_module = load_pia_ddpm_modules(repo_root)
    model, weights_key = load_pia_model(asset_summary["paths"]["checkpoint"], model_module, device=device)

    with np.load(asset_summary["paths"]["member_split"]) as split_payload:
        member_all = split_payload["mia_train_idxs"].tolist()
        nonmember_all = split_payload["mia_eval_idxs"].tolist()
    if (member_index_file is None) != (nonmember_index_file is None):
        raise ValueError("SimA explicit packet export requires both member_index_file and nonmember_index_file")

    selection_mode = "offset-slice"
    selected_packet_size: int | None = max(1, int(packet_size))
    if member_index_file is not None and nonmember_index_file is not None:
        selection_mode = "explicit-index-files"
        member_indices = _load_packet_indices_file(member_index_file)
        nonmember_indices = _load_packet_indices_file(nonmember_index_file)
        _validate_packet_indices(member_indices, member_all, "member")
        _validate_packet_indices(nonmember_indices, nonmember_all, "nonmember")
        if len(member_indices) == len(nonmember_indices):
            selected_packet_size = int(len(member_indices))
        else:
            selected_packet_size = None
    else:
        member_indices = member_all[int(member_offset) : int(member_offset) + int(selected_packet_size)]
        nonmember_indices = nonmember_all[int(nonmember_offset) : int(nonmember_offset) + int(selected_packet_size)]
        if len(member_indices) < int(selected_packet_size) or len(nonmember_indices) < int(selected_packet_size):
            raise ValueError("Requested SimA packet exceeds available member/non-member indices")

    member_loader = _build_pia_packet_loader(
        dataset=plan.dataset,
        dataset_dir=asset_summary["paths"]["dataset_dir"],
        packet_indices=member_indices,
        batch_size=batch_size,
    )
    nonmember_loader = _build_pia_packet_loader(
        dataset=plan.dataset,
        dataset_dir=asset_summary["paths"]["dataset_dir"],
        packet_indices=nonmember_indices,
        batch_size=batch_size,
    )

    alpha_bars = _alpha_bar_schedule()
    member_scores = _score_loader_for_timestep(
        model,
        member_loader,
        timestep=int(timestep),
        alpha_bars=alpha_bars,
        device=device,
        p_norm=p_norm,
        noise_seed=noise_seed,
    )
    nonmember_scores = _score_loader_for_timestep(
        model,
        nonmember_loader,
        timestep=int(timestep),
        alpha_bars=alpha_bars,
        device=device,
        p_norm=p_norm,
        noise_seed=noise_seed,
    )

    records: list[dict[str, Any]] = []
    for position, (split_index, score) in enumerate(
        zip(member_indices, member_scores.tolist(), strict=True)
    ):
        records.append(
            {
                "membership": "member",
                "packet_position": int(position),
                "split_index": int(split_index),
                "score": round(float(score), 6),
                "timestep": int(timestep),
                "p_norm": float(p_norm),
            }
        )
    for position, (split_index, score) in enumerate(
        zip(nonmember_indices, nonmember_scores.tolist(), strict=True)
    ):
        records.append(
            {
                "membership": "nonmember",
                "packet_position": int(position),
                "split_index": int(split_index),
                "score": round(float(score), 6),
                "timestep": int(timestep),
                "p_norm": float(p_norm),
            }
        )

    records_path = workspace_path / "sample_scores.jsonl"
    records_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )
    scores_path = workspace_path / "scores.json"
    scores_path.write_text(
        json.dumps(
            {
                "member_scores": [round(float(value), 6) for value in member_scores.tolist()],
                "nonmember_scores": [round(float(value), 6) for value in nonmember_scores.tolist()],
                "member_indices": [int(value) for value in member_indices],
                "nonmember_indices": [int(value) for value in nonmember_indices],
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    member_score_mean = float(member_scores.mean().item())
    nonmember_score_mean = float(nonmember_scores.mean().item())
    result = {
        "status": "ready",
        "track": "gray-box",
        "method": "sima",
        "paper": "SimA_arXiv2025",
        "mode": "packet-score-export",
        "device": "cpu",
        "gpu_release": "none",
        "admitted_change": "none",
        "provenance_status": provenance_status,
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
            "sample_scores": str(records_path),
            "scores": str(scores_path),
        },
        "runtime": {
            "repo_root": str(Path(repo_root)),
            "dataset_root": plan.data_root,
            "member_split_root": str(split_root),
            "weights_key": weights_key,
            "batch_size": int(batch_size),
            "selection_mode": selection_mode,
            "packet_size": int(selected_packet_size) if selected_packet_size is not None else None,
            "member_packet_size": int(len(member_indices)),
            "nonmember_packet_size": int(len(nonmember_indices)),
            "timestep": int(timestep),
            "p_norm": float(p_norm),
            "noise_seed": int(noise_seed),
        },
        "packet": {
            "member_indices": member_indices,
            "nonmember_indices": nonmember_indices,
            "member_score_mean": round(member_score_mean, 6),
            "nonmember_score_mean": round(nonmember_score_mean, 6),
            "member_control_score_gap": round(member_score_mean - nonmember_score_mean, 6),
        },
        "checks": {
            **asset_summary["checks"],
            "sample_scores_written": len(records),
            "packet_scores_generated": True,
            "device_cpu_only": True,
        },
        "notes": [
            "This scaffold exports packet-local SimA scores on a fixed member/non-member packet.",
            "It reuses the current DDPM/CIFAR10 runtime line without promoting SimA to challenger-ready status.",
            "It is a CPU-first bridge artifact for pairboard and support-fusion analysis, not a benchmark or GPU release.",
        ],
    }
    (workspace_path / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return result
