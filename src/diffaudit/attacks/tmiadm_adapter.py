"""Bounded TMIA-DM protocol probe on the current DDPM/CIFAR-10 asset line."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from diffaudit.attacks.pia_adapter import (
    PiaDefaults,
    _build_pia_subset_loader,
    _compute_auc,
    _compute_threshold_metrics,
    _dropout_is_active_for_timestep,
    _normalize_dropout_activation_schedule,
    _resolve_late_step_threshold,
    load_pia_model,
    prepare_pia_runtime,
    probe_pia_assets,
)
from diffaudit.config import AuditConfig


def _alpha_bar_schedule(
    defaults: PiaDefaults | None = None,
) -> torch.Tensor:
    config = defaults or PiaDefaults()
    betas = torch.linspace(config.beta_1, config.beta_T, config.T, dtype=torch.float32)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def _collect_eps_predictions_for_timestep(
    model: torch.nn.Module,
    loader,
    timestep: int,
    alpha_bars: torch.Tensor,
    device: str,
    noise_seed: int,
    dropout_activation_schedule: str = "off",
    late_step_threshold: int | None = None,
) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    alpha_bar_t = alpha_bars[int(timestep)].to(device=device)
    sqrt_alpha = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_bar_t)
    generator = torch.Generator(device=device if device.startswith("cuda") else "cpu")
    generator.manual_seed(int(noise_seed) + int(timestep))
    was_training = model.training
    if _dropout_is_active_for_timestep(
        dropout_activation_schedule,
        timestep=int(timestep),
        late_step_threshold=late_step_threshold,
    ):
        model.train()
    else:
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
            predictions.append(model(noisy, t=timestep_tensor).detach().cpu())
    model.train(was_training)
    return torch.cat(predictions, dim=0)


def _collect_eps_predictions_for_timesteps(
    model: torch.nn.Module,
    loader,
    timesteps: list[int],
    alpha_bars: torch.Tensor,
    device: str,
    noise_seed: int,
    dropout_activation_schedule: str = "off",
    late_step_threshold: int | None = None,
) -> torch.Tensor:
    timestep_predictions = [
        _collect_eps_predictions_for_timestep(
            model=model,
            loader=loader,
            timestep=timestep,
            alpha_bars=alpha_bars,
            device=device,
            noise_seed=noise_seed,
            dropout_activation_schedule=dropout_activation_schedule,
            late_step_threshold=late_step_threshold,
        )
        for timestep in timesteps
    ]
    return torch.stack(timestep_predictions, dim=1)


def _resolve_effective_timesteps(
    timesteps: list[int],
    timestep_jitter_radius: int = 0,
    noise_seed: int = 0,
    defaults: PiaDefaults | None = None,
) -> list[int]:
    if int(timestep_jitter_radius) <= 0:
        return [int(timestep) for timestep in timesteps]
    config = defaults or PiaDefaults()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(noise_seed))
    effective: list[int] = []
    for index, timestep in enumerate(timesteps):
        offset = int(
            torch.randint(
                low=-int(timestep_jitter_radius),
                high=int(timestep_jitter_radius) + 1,
                size=(1,),
                generator=generator,
            ).item()
        )
        jittered = max(0, min(config.T - 1, int(timestep) + offset))
        effective.append(int(jittered))
    return effective


def _apply_timestep_stride(
    timesteps: list[int],
    timestep_stride: int = 1,
) -> list[int]:
    stride = max(int(timestep_stride), 1)
    if stride <= 1:
        return [int(timestep) for timestep in timesteps]
    return [int(timestep) for index, timestep in enumerate(timesteps) if index % stride == 0]


def _zscore(scores: torch.Tensor) -> torch.Tensor:
    mean = scores.mean()
    std = scores.std(unbiased=False)
    if float(std.item()) == 0.0:
        return scores - mean
    return (scores - mean) / std


def _compute_tmiadm_family_scores(
    timestep_predictions: torch.Tensor,
    aggregation_p_norm: int | float = 2,
) -> dict[str, torch.Tensor]:
    if timestep_predictions.ndim < 3:
        raise ValueError("TMIA-DM protocol probe expects [samples, timesteps, ...] predictions")
    if timestep_predictions.shape[1] < 2:
        raise ValueError("TMIA-DM protocol probe requires at least two timesteps")

    flattened = timestep_predictions.flatten(start_dim=2)
    adjacent_deltas = flattened[:, 1:, :] - flattened[:, :-1, :]
    short_window_scores = -adjacent_deltas.abs().mean(dim=(1, 2))

    aggregated = flattened.mean(dim=1)
    long_window_scores = -torch.linalg.vector_norm(aggregated, ord=aggregation_p_norm, dim=1)

    fused_scores = (_zscore(short_window_scores) + _zscore(long_window_scores)) / 2.0
    return {
        "short_window": short_window_scores.detach().cpu(),
        "long_window": long_window_scores.detach().cpu(),
        "fused": fused_scores.detach().cpu(),
    }


def _compute_family_metrics(
    member_scores: torch.Tensor,
    nonmember_scores: torch.Tensor,
) -> dict[str, float]:
    return {
        "auc": round(_compute_auc(member_scores, nonmember_scores), 6),
        **_compute_threshold_metrics(
            member_scores.numpy(),
            nonmember_scores.numpy(),
        ),
        "member_score_mean": round(float(member_scores.mean().item()), 6),
        "nonmember_score_mean": round(float(nonmember_scores.mean().item()), 6),
    }


def _select_best_family(
    family_metrics: dict[str, dict[str, float]],
) -> tuple[str, dict[str, float]]:
    best_family = ""
    best_metrics: dict[str, float] | None = None
    family_priority = {"fused": 0, "short_window": 1, "long_window": 2}
    for family_name, metrics in family_metrics.items():
        if best_metrics is None:
            best_family = family_name
            best_metrics = metrics
            continue
        if (
            metrics["auc"] > best_metrics["auc"]
            or (
                metrics["auc"] == best_metrics["auc"]
                and family_priority.get(family_name, 99) < family_priority.get(best_family, 99)
            )
        ):
            best_family = family_name
            best_metrics = metrics
    if best_metrics is None:
        raise RuntimeError("TMIA-DM protocol probe produced no family metrics")
    return best_family, best_metrics


def run_tmiadm_protocol_probe(
    config: AuditConfig,
    workspace: str | Path,
    repo_root: str | Path = "external/PIA",
    member_split_root: str | Path | None = None,
    device: str = "cpu",
    max_samples: int | None = None,
    batch_size: int = 8,
    scan_timesteps: list[int] | tuple[int, ...] | None = None,
    aggregation_p_norm: int | float = 2,
    noise_seed: int = 0,
    stochastic_dropout_defense: bool = False,
    dropout_activation_schedule: str = "off",
    late_step_threshold: int | None = None,
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
    resolved_schedule = _normalize_dropout_activation_schedule(
        dropout_activation_schedule,
        stochastic_dropout_defense=stochastic_dropout_defense,
    )
    resolved_late_step_threshold = _resolve_late_step_threshold(
        attack_num=max(len(scan_timesteps or [20, 40, 60, 80, 100, 120]) - 1, 1),
        interval=max(
            int((scan_timesteps or [20, 40, 60, 80, 100, 120])[1] - (scan_timesteps or [20, 40, 60, 80, 100, 120])[0]),
            1,
        ) if len(scan_timesteps or [20, 40, 60, 80, 100, 120]) > 1 else 1,
        late_step_threshold=late_step_threshold,
    )
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

    timesteps = [int(t) for t in (scan_timesteps or [20, 40, 60, 80, 100, 120])]
    effective_timesteps = _resolve_effective_timesteps(
        timesteps,
        timestep_jitter_radius=timestep_jitter_radius,
        noise_seed=noise_seed,
    )
    effective_timesteps = _apply_timestep_stride(
        effective_timesteps,
        timestep_stride=timestep_stride,
    )
    if len(effective_timesteps) < 2:
        raise ValueError("TMIA-DM protocol probe requires at least two effective timesteps")
    alpha_bars = _alpha_bar_schedule()
    member_predictions = _collect_eps_predictions_for_timesteps(
        model=model,
        loader=member_loader,
        timesteps=effective_timesteps,
        alpha_bars=alpha_bars,
        device=device,
        noise_seed=noise_seed,
        dropout_activation_schedule=resolved_schedule,
        late_step_threshold=resolved_late_step_threshold,
    )
    nonmember_predictions = _collect_eps_predictions_for_timesteps(
        model=model,
        loader=nonmember_loader,
        timesteps=effective_timesteps,
        alpha_bars=alpha_bars,
        device=device,
        noise_seed=noise_seed,
        dropout_activation_schedule=resolved_schedule,
        late_step_threshold=resolved_late_step_threshold,
    )

    member_family_scores = _compute_tmiadm_family_scores(
        member_predictions,
        aggregation_p_norm=aggregation_p_norm,
    )
    nonmember_family_scores = _compute_tmiadm_family_scores(
        nonmember_predictions,
        aggregation_p_norm=aggregation_p_norm,
    )
    family_metrics = {
        family_name: _compute_family_metrics(member_scores, nonmember_family_scores[family_name])
        for family_name, member_scores in member_family_scores.items()
    }
    best_family, best_metrics = _select_best_family(family_metrics)

    family_scores_payload = {
        family_name: {
            "member_scores": [round(float(value), 6) for value in member_scores.tolist()],
            "nonmember_scores": [
                round(float(value), 6) for value in nonmember_family_scores[family_name].tolist()
            ],
            "metrics": family_metrics[family_name],
        }
        for family_name, member_scores in member_family_scores.items()
    }
    family_scores_path = workspace_path / "family-scores.json"
    family_scores_path.write_text(
        json.dumps(family_scores_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    result = {
        "status": "ready",
        "track": "gray-box",
        "method": "tmiadm",
        "paper": "TMIA-DM_arXiv2026",
        "mode": "protocol-probe",
        "device": device,
        "workspace": str(workspace_path),
        "workspace_name": workspace_path.name,
        "contract_stage": "target",
        "asset_grade": "single-machine-real-asset",
        "provenance_status": provenance_status,
        "evidence_level": "protocol-probe",
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
            "family_scores": str(family_scores_path),
        },
        "checks": {
            **asset_summary["checks"],
            "runtime_probe_ready": True,
            "multi_timestep_features_ready": True,
            "protocol_probe_completed": True,
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
            "aggregation_p_norm": float(aggregation_p_norm),
            "noise_seed": int(noise_seed),
            "elapsed_seconds": round(time.perf_counter() - started_at, 6),
        },
        "defense": {
            "name": (
                "stochastic-dropout"
                if stochastic_dropout_defense
                else "timestep-jitter"
                if int(timestep_jitter_radius) > 0
                else "temporal-striding"
                if int(timestep_stride) > 1
                else "none"
            ),
            "enabled": bool(
                stochastic_dropout_defense
                or int(timestep_jitter_radius) > 0
                or int(timestep_stride) > 1
            ),
            "dropout_activation_schedule": resolved_schedule,
            "late_step_threshold": int(resolved_late_step_threshold),
            "timestep_jitter_radius": int(timestep_jitter_radius),
            "timestep_stride": max(int(timestep_stride), 1),
        },
        "sample_count_per_split": int(len(member_indices)),
        "family_metrics": family_metrics,
        "best_family": best_family,
        "metrics": best_metrics,
        "notes": [
            "TMIA-DM protocol probe uses a short-window temporal-difference family over epsilon predictions.",
            "TMIA-DM protocol probe also uses a long-window temporal aggregation family over the same timesteps.",
            "This is a bounded local protocol probe, not yet a paper-faithful reproduction or promoted challenger verdict.",
        ],
    }
    (workspace_path / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return result
