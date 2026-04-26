"""Runtime helpers for the official PIA DDPM implementation."""

from __future__ import annotations

import importlib.util
import json
import math
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets as tv_datasets
from torchvision import transforms as tv_transforms

from diffaudit.attacks.pia import (
    PiaPlan,
    build_pia_plan,
    probe_pia_assets,
    validate_pia_workspace,
)
from diffaudit.config import (
    AssetConfig,
    AttackConfig,
    AuditConfig,
    ReportConfig,
    TaskConfig,
)


@dataclass(frozen=True)
class PiaDefaults:
    T: int = 1000
    ch: int = 128
    ch_mult: tuple[int, ...] = (1, 2, 2, 2)
    attn: tuple[int, ...] = (1,)
    num_res_blocks: int = 2
    dropout: float = 0.1
    beta_1: float = 0.0001
    beta_T: float = 0.02
    image_size: int = 32


@dataclass(frozen=True)
class PiaAdapterContext:
    plan: PiaPlan
    components_module: ModuleType
    model_module: ModuleType
    checkpoint_path: str
    weights_key: str


VALID_DROPOUT_ACTIVATION_SCHEDULES = {
    "off",
    "all_steps",
    "late_steps_only",
}


def _load_module_from_path(module_name: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pia_ddpm_modules(repo_root: str | Path) -> tuple[ModuleType, ModuleType]:
    repo_path = Path(repo_root)
    components_module = _load_module_from_path(
        f"diffaudit_pia_components_{abs(hash(repo_path.resolve()))}",
        repo_path / "DDPM" / "components.py",
    )
    model_module = _load_module_from_path(
        f"diffaudit_pia_model_{abs(hash(repo_path.resolve()))}",
        repo_path / "DDPM" / "model.py",
    )
    return components_module, model_module


def build_pia_unet(model_module: ModuleType, defaults: PiaDefaults | None = None) -> torch.nn.Module:
    config = defaults or PiaDefaults()
    return model_module.UNet(
        T=config.T,
        ch=config.ch,
        ch_mult=list(config.ch_mult),
        attn=list(config.attn),
        num_res_blocks=config.num_res_blocks,
        dropout=config.dropout,
    )


def resolve_pia_weights_key(checkpoint: dict[str, Any]) -> str:
    if "ema_model" in checkpoint:
        return "ema_model"
    if "net_model" in checkpoint:
        return "net_model"
    raise KeyError("PIA checkpoint must contain 'ema_model' or 'net_model'")


def load_pia_model(
    checkpoint_path: str | Path,
    model_module: ModuleType,
    device: str = "cpu",
    defaults: PiaDefaults | None = None,
) -> tuple[torch.nn.Module, str]:
    model = build_pia_unet(model_module, defaults=defaults)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("PIA checkpoint must be a dict-like torch checkpoint")
    weights_key = resolve_pia_weights_key(checkpoint)
    weights = checkpoint[weights_key]
    state_dict = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in weights.items()
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, weights_key


def bootstrap_pia_smoke_assets(
    target_dir: str | Path,
    repo_root: str | Path,
    defaults: PiaDefaults | None = None,
) -> dict[str, Any]:
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    _, model_module = load_pia_ddpm_modules(repo_root)
    model = build_pia_unet(model_module, defaults=defaults)
    state_dict = model.state_dict()
    checkpoint = {
        "ema_model": state_dict,
        "net_model": state_dict,
    }
    checkpoint_path = target_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    return {
        "checkpoint_path": str(checkpoint_path),
        "weights_key": "ema_model",
    }


def _build_pia_eps_getter(
    components_module: ModuleType,
    model: torch.nn.Module,
    dropout_activation_schedule: str = "off",
    late_step_threshold: int | None = None,
    epsilon_output_noise_std: float = 0.0,
    epsilon_precision_bins: int | None = None,
):
    def precision_throttle(eps_prediction: torch.Tensor) -> torch.Tensor:
        if epsilon_precision_bins is None:
            return eps_prediction
        bins = int(epsilon_precision_bins)
        if bins <= 1:
            raise ValueError("epsilon_precision_bins must be greater than 1")
        clipped = torch.clamp(eps_prediction, -1.0, 1.0)
        scaled = (clipped + 1.0) * ((bins - 1) / 2.0)
        return torch.round(scaled) * (2.0 / (bins - 1)) - 1.0

    class RuntimeEpsGetter(components_module.EpsGetter):
        def __call__(
            self,
            xt: torch.Tensor,
            condition: torch.Tensor | None = None,
            noise_level: torch.Tensor | None = None,
            t: int | None = None,
        ) -> torch.Tensor:
            del condition, noise_level
            timestep_value = int(t or 0)
            if _dropout_is_active_for_timestep(
                dropout_activation_schedule,
                timestep=timestep_value,
                late_step_threshold=late_step_threshold,
            ):
                self.model.train()
            else:
                self.model.eval()
            timestep = torch.ones([xt.shape[0]], device=xt.device, dtype=torch.long) * int(t or 0)
            eps_prediction = self.model(xt, t=timestep)
            eps_prediction = precision_throttle(eps_prediction)
            if float(epsilon_output_noise_std) > 0.0:
                eps_prediction = eps_prediction + torch.randn_like(eps_prediction) * float(epsilon_output_noise_std)
            return eps_prediction

    return RuntimeEpsGetter(model)


def _build_pia_attacker(
    components_module: ModuleType,
    model: torch.nn.Module,
    attacker_name: str,
    attack_num: int,
    interval: int,
    device: str,
    dropout_activation_schedule: str = "off",
    late_step_threshold: int | None = None,
    epsilon_output_noise_std: float = 0.0,
    epsilon_precision_bins: int | None = None,
    defaults: PiaDefaults | None = None,
):
    config = defaults or PiaDefaults()
    attacker_cls = getattr(components_module, attacker_name, None)
    if attacker_cls is None:
        raise ValueError(f"PIA attacker class not found: {attacker_name}")
    betas = torch.linspace(config.beta_1, config.beta_T, config.T, device=device)
    eps_getter = _build_pia_eps_getter(
        components_module,
        model,
        dropout_activation_schedule=dropout_activation_schedule,
        late_step_threshold=late_step_threshold,
        epsilon_output_noise_std=epsilon_output_noise_std,
        epsilon_precision_bins=epsilon_precision_bins,
    )
    return attacker_cls(
        betas,
        interval,
        attack_num,
        eps_getter,
        normalize=lambda x: x * 2 - 1,
    )


def prepare_pia_runtime(
    config: AuditConfig,
    repo_root: str | Path,
    member_split_root: str | Path | None = None,
    device: str = "cpu",
) -> PiaAdapterContext:
    plan = build_pia_plan(config)
    validate_pia_workspace(repo_root)
    split_root = Path(member_split_root) if member_split_root else Path(repo_root) / "DDPM"
    summary = probe_pia_assets(
        dataset=plan.dataset,
        dataset_root=plan.data_root,
        model_dir=plan.model_dir,
        member_split_root=split_root,
    )
    if summary["status"] != "ready":
        missing_description = summary["missing_description"] or "PIA assets"
        raise FileNotFoundError(f"PIA assets not ready: {missing_description}")

    components_module, model_module = load_pia_ddpm_modules(repo_root)
    checkpoint_path = summary["paths"]["checkpoint"]
    model, weights_key = load_pia_model(checkpoint_path, model_module, device=device)
    _build_pia_attacker(
        components_module=components_module,
        model=model,
        attacker_name=plan.attacker_name,
        attack_num=plan.attack_num,
        interval=plan.interval,
        device=device,
    )
    return PiaAdapterContext(
        plan=plan,
        components_module=components_module,
        model_module=model_module,
        checkpoint_path=checkpoint_path,
        weights_key=weights_key,
    )


def probe_pia_runtime(
    config: AuditConfig,
    repo_root: str | Path,
    member_split_root: str | Path | None = None,
    device: str = "cpu",
) -> tuple[int, dict[str, Any]]:
    try:
        plan = build_pia_plan(config)
        validate_pia_workspace(repo_root)
        split_root = Path(member_split_root) if member_split_root else Path(repo_root) / "DDPM"
        summary = probe_pia_assets(
            dataset=plan.dataset,
            dataset_root=plan.data_root,
            model_dir=plan.model_dir,
            member_split_root=split_root,
        )
        if summary["status"] != "ready":
            return 1, {
                "status": "blocked",
                "repo_root": str(repo_root),
                "checks": summary["checks"],
                "paths": summary["paths"],
                "missing": summary["missing"],
                "missing_keys": summary["missing_keys"],
                "missing_items": summary["missing_items"],
                "missing_description": summary["missing_description"],
            }

        components_module, model_module = load_pia_ddpm_modules(repo_root)
        model, weights_key = load_pia_model(summary["paths"]["checkpoint"], model_module, device=device)
        attacker = _build_pia_attacker(
            components_module=components_module,
            model=model,
            attacker_name=plan.attacker_name,
            attack_num=plan.attack_num,
            interval=plan.interval,
            device=device,
        )
        preview_batch = torch.rand(1, 3, 32, 32, device=device)
        preview_scores = attacker(preview_batch)
    except (FileNotFoundError, ValueError, KeyError, TypeError, ImportError) as exc:
        return 1, {
            "status": "blocked",
            "error": str(exc),
            "repo_root": str(repo_root),
            "device": device,
        }
    except Exception as exc:
        return 2, {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "repo_root": str(repo_root),
            "device": device,
        }

    return 0, {
        "status": "ready",
        "track": "gray-box",
        "method": "pia",
        "paper": "PIA_ICLR2024",
        "mode": "runtime-probe",
        "device": device,
        "repo_root": str(Path(repo_root)),
        "dataset": plan.dataset,
        "model_dir": plan.model_dir,
        "attack_num": plan.attack_num,
        "interval": plan.interval,
        "attacker_name": plan.attacker_name,
        "checks": {
            **summary["checks"],
            "components_loaded": hasattr(components_module, plan.attacker_name),
            "model_loaded": True,
            "attacker_instantiated": True,
            "preview_forward": True,
        },
        "paths": summary["paths"],
        "weights_key": weights_key,
        "preview_score_shape": list(preview_scores.shape),
    }


def _normalize_dropout_activation_schedule(
    dropout_activation_schedule: str,
    stochastic_dropout_defense: bool,
) -> str:
    schedule = str(dropout_activation_schedule).strip().lower() or "off"
    if schedule not in VALID_DROPOUT_ACTIVATION_SCHEDULES:
        allowed = ", ".join(sorted(VALID_DROPOUT_ACTIVATION_SCHEDULES))
        raise ValueError(f"Unsupported dropout activation schedule: {schedule}. Expected one of {allowed}")
    if not stochastic_dropout_defense:
        return "off"
    if schedule == "off":
        return "all_steps"
    return schedule


def _resolve_late_step_threshold(
    attack_num: int,
    interval: int,
    late_step_threshold: int | None,
) -> int:
    if late_step_threshold is not None and late_step_threshold > 0:
        return int(late_step_threshold)
    return max(((attack_num + 1) // 2) * interval, interval)


def _dropout_is_active_for_timestep(
    dropout_activation_schedule: str,
    timestep: int,
    late_step_threshold: int | None,
) -> bool:
    if dropout_activation_schedule == "all_steps":
        return True
    if dropout_activation_schedule == "late_steps_only":
        return timestep >= int(late_step_threshold or 0)
    return False


def _apply_input_defense(
    batch: torch.Tensor,
    input_gaussian_blur_sigma: float = 0.0,
) -> torch.Tensor:
    if batch.numel() == 0:
        return batch
    if float(input_gaussian_blur_sigma) > 0.0:
        return tv_transforms.functional.gaussian_blur(
            batch,
            kernel_size=[3, 3],
            sigma=float(input_gaussian_blur_sigma),
        )
    return batch


def _score_batch_once(
    attacker,
    batch: torch.Tensor,
    device: str,
    input_gaussian_blur_sigma: float = 0.0,
) -> torch.Tensor:
    defended_batch = _apply_input_defense(
        batch,
        input_gaussian_blur_sigma=input_gaussian_blur_sigma,
    )
    batch_scores = attacker(defended_batch.to(device)).mean(dim=0)
    return (-batch_scores).detach().cpu()


def _score_loader(
    attacker,
    loader,
    device: str,
    adaptive_query_repeats: int,
    input_gaussian_blur_sigma: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    aggregated_scores: list[torch.Tensor] = []
    std_scores: list[torch.Tensor] = []
    single_scores: list[torch.Tensor] = []
    with torch.no_grad():
        for batch, _ in loader:
            repeated_scores = [
                _score_batch_once(
                    attacker,
                    batch,
                    device=device,
                    input_gaussian_blur_sigma=input_gaussian_blur_sigma,
                )
                for _ in range(max(adaptive_query_repeats, 1))
            ]
            stacked_scores = torch.stack(repeated_scores)
            single_scores.append(stacked_scores[0])
            aggregated_scores.append(stacked_scores.mean(dim=0))
            std_scores.append(stacked_scores.std(dim=0, unbiased=False))
    return (
        torch.cat(single_scores),
        torch.cat(aggregated_scores),
        torch.cat(std_scores),
    )


def _collect_loader_batches(
    loader,
) -> torch.Tensor:
    batches: list[torch.Tensor] = []
    for batch, _ in loader:
        batches.append(batch)
    if not batches:
        return torch.empty(0)
    return torch.cat(batches, dim=0)


def _predict_surrogate_images(
    model: torch.nn.Module,
    batches: torch.Tensor,
    device: str,
    dropout_activation_schedule: str,
    late_step_threshold: int | None,
    epsilon_output_noise_std: float = 0.0,
    epsilon_precision_bins: int | None = None,
    input_gaussian_blur_sigma: float = 0.0,
    surrogate_batch_size: int = 64,
) -> torch.Tensor:
    def precision_throttle(eps_prediction: torch.Tensor) -> torch.Tensor:
        if epsilon_precision_bins is None:
            return eps_prediction
        bins = int(epsilon_precision_bins)
        if bins <= 1:
            raise ValueError("epsilon_precision_bins must be greater than 1")
        clipped = torch.clamp(eps_prediction, -1.0, 1.0)
        scaled = (clipped + 1.0) * ((bins - 1) / 2.0)
        return torch.round(scaled) * (2.0 / (bins - 1)) - 1.0

    if batches.numel() == 0:
        return torch.empty_like(batches)
    was_training = model.training
    if _dropout_is_active_for_timestep(
        dropout_activation_schedule,
        timestep=0,
        late_step_threshold=late_step_threshold,
    ):
        model.train()
    else:
        model.eval()
    surrogate_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, batches.shape[0], max(surrogate_batch_size, 1)):
            chunk = batches[start : start + max(surrogate_batch_size, 1)]
            defended_chunk = _apply_input_defense(
                chunk,
                input_gaussian_blur_sigma=input_gaussian_blur_sigma,
            )
            normalized = defended_chunk.to(device) * 2 - 1
            timestep = torch.zeros(normalized.shape[0], device=device, dtype=torch.long)
            eps_prediction = model(normalized, t=timestep)
            eps_prediction = precision_throttle(eps_prediction)
            if float(epsilon_output_noise_std) > 0.0:
                eps_prediction = eps_prediction + torch.randn_like(eps_prediction) * float(epsilon_output_noise_std)
            surrogate = torch.clamp((normalized - eps_prediction + 1.0) / 2.0, 0.0, 1.0)
            surrogate_chunks.append(surrogate.detach().cpu())
    model.train(was_training)
    return torch.cat(surrogate_chunks, dim=0)


def _frechet_distance_from_features(
    lhs_features: np.ndarray,
    rhs_features: np.ndarray,
) -> float:
    lhs_mu = lhs_features.mean(axis=0)
    rhs_mu = rhs_features.mean(axis=0)
    lhs_cov = np.cov(lhs_features, rowvar=False)
    rhs_cov = np.cov(rhs_features, rowvar=False)
    covariance_product = lhs_cov @ rhs_cov
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_product)
    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
    covariance_sqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    mean_delta = lhs_mu - rhs_mu
    fid = mean_delta @ mean_delta + np.trace(lhs_cov + rhs_cov - 2 * covariance_sqrt)
    return float(max(fid, 0.0))


def _surrogate_feature_vectors(images: torch.Tensor) -> np.ndarray:
    per_channel_mean = images.mean(dim=(2, 3))
    per_channel_std = images.std(dim=(2, 3), unbiased=False)
    return torch.cat([per_channel_mean, per_channel_std], dim=1).cpu().numpy()


def _compute_surrogate_quality_metrics(
    reference_images: torch.Tensor,
    active_images: torch.Tensor,
    baseline_images: torch.Tensor | None = None,
) -> dict[str, Any]:
    reference_name = "baseline_surrogate" if baseline_images is not None else "input_batch"
    comparison_reference = baseline_images if baseline_images is not None else reference_images
    comparison_reference = comparison_reference.to(active_images.dtype)
    lpips_value = torch.sqrt(((active_images - comparison_reference) ** 2).mean(dim=(1, 2, 3))).mean()
    fid_value = _frechet_distance_from_features(
        _surrogate_feature_vectors(active_images),
        _surrogate_feature_vectors(comparison_reference),
    )
    return {
        "suite": "pia-runtime-surrogate-v1",
        "official": False,
        "reference": reference_name,
        "metrics": {
            "fid": {
                "value": round(float(fid_value), 6),
                "official": False,
                "feature_space": "rgb-channel-moments",
            },
            "lpips": {
                "value": round(float(lpips_value.item()), 6),
                "official": False,
                "backbone": "none",
                "distance": "root-mean-square",
            },
            "inception_score": {
                "value": None,
                "official": False,
                "status": "not_computed",
                "reason": "no classifier-backed surrogate quality model is wired into the current PIA runtime path",
            },
        },
        "notes": [
            "Quality metrics are surrogate diagnostics over runtime batches, not paper-grade image generation metrics.",
            "FID uses rgb-channel-moment features and LPIPS uses RMS image drift against the chosen reference batch.",
        ],
    }


def _load_pia_preview_batches(
    dataset: str,
    dataset_dir: str | Path,
    member_split_path: str | Path,
    preview_batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dataset.upper() != "CIFAR10":
        raise ValueError(f"PIA runtime preview currently supports CIFAR10 only: {dataset}")

    split_payload = np.load(member_split_path)
    member_indices = split_payload["mia_train_idxs"].tolist()
    nonmember_indices = split_payload["mia_eval_idxs"].tolist()
    if not member_indices or not nonmember_indices:
        raise ValueError("PIA member split must contain non-empty member and non-member indices")

    transform = tv_transforms.Compose([tv_transforms.ToTensor()])
    dataset_obj = tv_datasets.CIFAR10(
        root=str(dataset_dir),
        train=True,
        download=False,
        transform=transform,
    )
    member_subset = Subset(dataset_obj, member_indices)
    nonmember_subset = Subset(dataset_obj, nonmember_indices)
    member_loader = DataLoader(
        member_subset,
        batch_size=min(preview_batch_size, len(member_subset)),
        shuffle=False,
    )
    nonmember_loader = DataLoader(
        nonmember_subset,
        batch_size=min(preview_batch_size, len(nonmember_subset)),
        shuffle=False,
    )
    member_batch = next(iter(member_loader))[0].to(device)
    nonmember_batch = next(iter(nonmember_loader))[0].to(device)
    return member_batch, nonmember_batch


def probe_pia_runtime_preview(
    config: AuditConfig,
    repo_root: str | Path,
    member_split_root: str | Path | None = None,
    device: str = "cpu",
    preview_batch_size: int = 2,
) -> tuple[int, dict[str, Any]]:
    try:
        plan = build_pia_plan(config)
        validate_pia_workspace(repo_root)
        split_root = Path(member_split_root) if member_split_root else Path(repo_root) / "DDPM"
        summary = probe_pia_assets(
            dataset=plan.dataset,
            dataset_root=plan.data_root,
            model_dir=plan.model_dir,
            member_split_root=split_root,
        )
        if summary["status"] != "ready":
            return 1, {
                "status": "blocked",
                "repo_root": str(repo_root),
                "checks": summary["checks"],
                "paths": summary["paths"],
                "missing": summary["missing"],
                "missing_keys": summary["missing_keys"],
                "missing_items": summary["missing_items"],
                "missing_description": summary["missing_description"],
            }

        components_module, model_module = load_pia_ddpm_modules(repo_root)
        model, weights_key = load_pia_model(summary["paths"]["checkpoint"], model_module, device=device)
        attacker = _build_pia_attacker(
            components_module=components_module,
            model=model,
            attacker_name=plan.attacker_name,
            attack_num=plan.attack_num,
            interval=plan.interval,
            device=device,
        )
        member_batch, nonmember_batch = _load_pia_preview_batches(
            dataset=plan.dataset,
            dataset_dir=summary["paths"]["dataset_dir"],
            member_split_path=summary["paths"]["member_split"],
            preview_batch_size=preview_batch_size,
            device=device,
        )
        member_scores = attacker(member_batch)
        nonmember_scores = attacker(nonmember_batch)
    except (FileNotFoundError, ValueError, KeyError, TypeError, ImportError) as exc:
        return 1, {
            "status": "blocked",
            "error": str(exc),
            "repo_root": str(repo_root),
            "device": device,
        }
    except Exception as exc:
        return 2, {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "repo_root": str(repo_root),
            "device": device,
        }

    return 0, {
        "status": "ready",
        "track": "gray-box",
        "method": "pia",
        "paper": "PIA_ICLR2024",
        "mode": "runtime-preview",
        "device": device,
        "repo_root": str(Path(repo_root)),
        "dataset": plan.dataset,
        "model_dir": plan.model_dir,
        "attack_num": plan.attack_num,
        "interval": plan.interval,
        "attacker_name": plan.attacker_name,
        "preview_batch_size": preview_batch_size,
        "checks": {
            **summary["checks"],
            "components_loaded": hasattr(components_module, plan.attacker_name),
            "model_loaded": True,
            "attacker_instantiated": True,
            "member_preview_loaded": True,
            "nonmember_preview_loaded": True,
            "member_preview_forward": True,
            "nonmember_preview_forward": True,
        },
        "paths": summary["paths"],
        "weights_key": weights_key,
        "member_batch_shape": list(member_batch.shape),
        "nonmember_batch_shape": list(nonmember_batch.shape),
        "member_score_shape": list(member_scores.shape),
        "nonmember_score_shape": list(nonmember_scores.shape),
        "member_score_mean": float(member_scores.mean().item()),
        "nonmember_score_mean": float(nonmember_scores.mean().item()),
    }


def _compute_auc(member_scores: torch.Tensor, nonmember_scores: torch.Tensor) -> float:
    member = member_scores.detach().cpu().flatten()
    nonmember = nonmember_scores.detach().cpu().flatten()
    wins = (member[:, None] > nonmember[None, :]).float()
    ties = (member[:, None] == nonmember[None, :]).float() * 0.5
    return float((wins + ties).mean().item())


def _compute_threshold_metrics(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
) -> dict[str, float]:
    scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate(
        [
            np.ones(member_scores.shape[0], dtype=np.int64),
            np.zeros(nonmember_scores.shape[0], dtype=np.int64),
        ]
    )
    thresholds = np.unique(scores)[::-1]
    if thresholds.size == 0:
        thresholds = np.asarray([0.0], dtype=float)

    best_asr = -1.0
    best_threshold = float(thresholds[0])
    best_tpr_at_1pct = 0.0
    best_tpr_at_0_1pct = 0.0
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(np.int64)
        tp = int(((predictions == 1) & (labels == 1)).sum())
        tn = int(((predictions == 0) & (labels == 0)).sum())
        fp = int(((predictions == 1) & (labels == 0)).sum())
        fn = int(((predictions == 0) & (labels == 1)).sum())
        asr = float((tp + tn) / labels.shape[0])
        if asr > best_asr:
            best_asr = asr
            best_threshold = float(threshold)
        tpr = float(tp / (tp + fn)) if (tp + fn) else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
        if fpr <= 0.01:
            best_tpr_at_1pct = max(best_tpr_at_1pct, tpr)
        if fpr <= 0.001:
            best_tpr_at_0_1pct = max(best_tpr_at_0_1pct, tpr)

    return {
        "asr": round(best_asr, 6),
        "threshold": round(best_threshold, 6),
        "tpr_at_1pct_fpr": round(best_tpr_at_1pct, 6),
        "tpr_at_0_1pct_fpr": round(best_tpr_at_0_1pct, 6),
    }


def _select_preview_indices(
    split_indices: list[int],
    max_samples: int | None,
) -> list[int]:
    if max_samples is None or max_samples <= 0:
        return list(split_indices)
    return list(split_indices[: max_samples])


def _build_pia_subset_loader(
    dataset: str,
    dataset_dir: str | Path,
    member_split_path: str | Path,
    batch_size: int,
    max_samples: int | None,
    membership: str,
):
    if dataset.upper() != "CIFAR10":
        raise ValueError(f"PIA runtime mainline currently supports CIFAR10 only: {dataset}")

    split_payload = np.load(member_split_path)
    if membership == "member":
        raw_indices = split_payload["mia_train_idxs"].tolist()
    else:
        raw_indices = split_payload["mia_eval_idxs"].tolist()
    selected_indices = _select_preview_indices(raw_indices, max_samples=max_samples)
    if not selected_indices:
        raise ValueError(f"PIA split has no samples for membership={membership}")

    transform = tv_transforms.Compose([tv_transforms.ToTensor()])
    dataset_obj = tv_datasets.CIFAR10(
        root=str(dataset_dir),
        train=True,
        download=False,
        transform=transform,
    )
    subset = Subset(dataset_obj, selected_indices)
    loader = DataLoader(
        subset,
        batch_size=min(batch_size, len(subset)),
        shuffle=False,
    )
    return loader, selected_indices


def _build_pia_packet_loader(
    dataset: str,
    dataset_dir: str | Path,
    packet_indices: list[int],
    batch_size: int,
):
    if dataset.upper() != "CIFAR10":
        raise ValueError(f"PIA packet export currently supports CIFAR10 only: {dataset}")
    if not packet_indices:
        raise ValueError("PIA packet export requires at least one packet index")

    transform = tv_transforms.Compose([tv_transforms.ToTensor()])
    dataset_obj = tv_datasets.CIFAR10(
        root=str(dataset_dir),
        train=True,
        download=False,
        transform=transform,
    )
    subset = Subset(dataset_obj, packet_indices)
    loader = DataLoader(
        subset,
        batch_size=min(batch_size, len(subset)),
        shuffle=False,
    )
    return loader


def _load_packet_indices_file(path: str | Path) -> list[int]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"PIA packet index file is empty: {path}")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        tokens = [token for token in re.split(r"[\s,]+", text) if token]
        if not tokens:
            raise ValueError(f"PIA packet index file does not contain any indices: {path}")
        return [int(token) for token in tokens]

    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"PIA packet index file does not contain any indices: {path}")
        return [int(value) for value in payload]
    if isinstance(payload, dict) and isinstance(payload.get("indices"), list):
        if not payload["indices"]:
            raise ValueError(f"PIA packet index file does not contain any indices: {path}")
        return [int(value) for value in payload["indices"]]
    raise ValueError(f"Unsupported PIA packet index file format: {path}")


def _validate_packet_indices(
    requested_indices: list[int],
    allowed_indices: list[int],
    membership: str,
) -> None:
    allowed = {int(value) for value in allowed_indices}
    invalid = [int(value) for value in requested_indices if int(value) not in allowed]
    if invalid:
        raise ValueError(
            f"PIA explicit {membership} packet includes indices outside the canonical {membership} split: {invalid[:5]}"
        )


def _extract_pia_model_prediction(model_output: Any) -> torch.Tensor:
    if hasattr(model_output, "sample"):
        prediction = model_output.sample
    elif isinstance(model_output, tuple):
        prediction = model_output[0]
    else:
        prediction = model_output
    if not isinstance(prediction, torch.Tensor):
        raise TypeError(f"Unexpected PIA model prediction type: {type(prediction).__name__}")
    return prediction


def _normalize_channel_dim(activation: torch.Tensor, channel_dim: int) -> int:
    normalized = int(channel_dim)
    if normalized < 0:
        normalized += int(activation.ndim)
    if normalized < 0 or normalized >= int(activation.ndim):
        raise ValueError(
            f"Invalid channel_dim={channel_dim} for activation shape={tuple(int(dim) for dim in activation.shape)}"
        )
    return normalized


def _channelwise_profile_for_dim(
    activation: torch.Tensor,
    channel_dim: int,
) -> torch.Tensor:
    if activation.ndim == 0:
        return activation.reshape(1)
    normalized_dim = _normalize_channel_dim(activation, channel_dim)
    moved = torch.movedim(activation, normalized_dim, -1)
    channel_count = int(moved.shape[-1])
    return moved.reshape(-1, channel_count).mean(dim=0)


def _apply_channel_mask_for_dim(
    activation: torch.Tensor,
    channel_indices: list[int],
    alpha: float,
    channel_dim: int,
) -> torch.Tensor:
    masked = activation.clone()
    if not channel_indices:
        return masked
    normalized_dim = _normalize_channel_dim(masked, channel_dim)
    index = [slice(None)] * int(masked.ndim)
    index[normalized_dim] = channel_indices
    masked[tuple(index)] = masked[tuple(index)] * float(alpha)
    return masked


def _select_mask_indices(
    canary_profile: torch.Tensor,
    control_profile: torch.Tensor,
    mask_kind: str,
    k: int,
    random_seed: int,
) -> list[int]:
    channel_count = int(canary_profile.shape[0])
    selected_k = max(1, min(int(k), channel_count))
    delta = torch.abs(canary_profile - control_profile)
    if mask_kind == "top_abs_delta_k":
        return torch.argsort(delta, descending=True)[:selected_k].tolist()
    if mask_kind == "bottom_abs_delta_k":
        return torch.argsort(delta, descending=False)[:selected_k].tolist()
    if mask_kind == "random_k_seeded":
        rng = np.random.default_rng(int(random_seed))
        return sorted(int(idx) for idx in rng.choice(channel_count, size=selected_k, replace=False).tolist())
    raise ValueError(f"Unsupported mask kind: {mask_kind}")


def _resolve_pia_alias_module(
    model: torch.nn.Module,
    alias_selector: str,
) -> torch.nn.Module:
    modules = dict(model.named_modules())
    if alias_selector not in modules:
        raise KeyError(f"PIA alias selector not found in model: {alias_selector}")
    return modules[alias_selector]


def _run_pia_alias_forward(
    model: torch.nn.Module,
    alias_selector: str,
    batch: torch.Tensor,
    channel_dim: int,
    timestep: int = 0,
    channel_indices: list[int] | None = None,
    alpha: float = 1.0,
) -> dict[str, Any]:
    captured: dict[str, Any] = {
        "hook_hits": 0,
    }
    alias_module = _resolve_pia_alias_module(model, alias_selector)
    alias_weight = getattr(alias_module, "weight", None)
    alias_weight_shape = list(alias_weight.shape) if isinstance(alias_weight, torch.Tensor) else None
    model_device = next(model.parameters()).device
    normalized_batch = (batch.to(model_device) * 2.0) - 1.0
    timestep_tensor = torch.full(
        (normalized_batch.shape[0],),
        int(timestep),
        device=model_device,
        dtype=torch.long,
    )

    def _hook(_: torch.nn.Module, __: tuple[torch.Tensor, ...], output: Any):
        tensor = output[0] if isinstance(output, tuple) else output
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Unexpected alias output type: {type(tensor).__name__}")
        captured["pre_activation"] = tensor.detach().cpu()
        captured["hook_hits"] = int(captured["hook_hits"]) + 1
        if channel_indices:
            masked = _apply_channel_mask_for_dim(
                tensor,
                channel_indices=channel_indices,
                alpha=alpha,
                channel_dim=channel_dim,
            )
            captured["post_activation"] = masked.detach().cpu()
            if isinstance(output, tuple):
                return (masked, *output[1:])
            return masked
        captured["post_activation"] = tensor.detach().cpu()
        return output

    handle = alias_module.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model_output = model(normalized_batch, t=timestep_tensor)
    finally:
        handle.remove()

    if "pre_activation" not in captured or "post_activation" not in captured:
        raise RuntimeError(f"PIA alias hook did not capture output for selector: {alias_selector}")
    prediction = _extract_pia_model_prediction(model_output).detach().cpu()
    return {
        "pre_activation": captured["pre_activation"],
        "post_activation": captured["post_activation"],
        "prediction": prediction,
        "hook_hits": int(captured["hook_hits"]),
        "alias_weight_shape": alias_weight_shape,
    }


def _score_loader_with_alias_probe(
    attacker,
    model: torch.nn.Module,
    loader,
    alias_selector: str,
    channel_dim: int,
    device: str,
    adaptive_query_repeats: int,
    channel_indices: list[int] | None = None,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "hook_hits": 0,
        "alias_activation_shape": None,
        "alias_weight_shape": None,
    }
    alias_module = _resolve_pia_alias_module(model, alias_selector)
    alias_weight = getattr(alias_module, "weight", None)
    if isinstance(alias_weight, torch.Tensor):
        metadata["alias_weight_shape"] = list(alias_weight.shape)

    def _hook(_: torch.nn.Module, __: tuple[torch.Tensor, ...], output: Any):
        tensor = output[0] if isinstance(output, tuple) else output
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Unexpected alias output type: {type(tensor).__name__}")
        metadata["hook_hits"] = int(metadata["hook_hits"]) + 1
        if metadata["alias_activation_shape"] is None:
            metadata["alias_activation_shape"] = list(tensor.shape)
        if channel_indices:
            masked = _apply_channel_mask_for_dim(
                tensor,
                channel_indices=channel_indices,
                alpha=alpha,
                channel_dim=channel_dim,
            )
            if isinstance(output, tuple):
                return (masked, *output[1:])
            return masked
        return output

    handle = alias_module.register_forward_hook(_hook)
    try:
        score, adaptive_score, score_std = _score_loader(
            attacker,
            loader,
            device=device,
            adaptive_query_repeats=adaptive_query_repeats,
        )
    finally:
        handle.remove()
    return score, adaptive_score, score_std, metadata


def export_pia_translated_alias_probe(
    config: AuditConfig,
    workspace: str | Path,
    repo_root: str | Path = "external/PIA",
    member_split_root: str | Path | None = None,
    device: str = "cpu",
    member_index: int = 0,
    nonmember_index: int = 0,
    batch_size: int = 1,
    adaptive_query_repeats: int = 1,
    alias_selector: str = "middleblocks.0.attn.proj_v",
    translated_from: str = "mid_block.attentions.0.to_v",
    channel_dim: int = 1,
    mask_kind: str = "top_abs_delta_k",
    k: int = 8,
    alpha: float = 0.5,
    mask_seed: int = 0,
    alias_timestep: int = 0,
    provenance_status: str = "workspace-verified",
) -> dict[str, Any]:
    if device.lower() != "cpu":
        raise ValueError("PIA translated alias probe is CPU-first and does not authorize GPU use.")

    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    tensors_root = workspace_path / "tensors"
    tensors_root.mkdir(parents=True, exist_ok=True)
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
            "method": "pia",
            "paper": "PIA_ICLR2024",
            "mode": "translated-alias-probe",
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

    components_module, model_module = load_pia_ddpm_modules(repo_root)
    model, weights_key = load_pia_model(asset_summary["paths"]["checkpoint"], model_module, device=device)
    attacker = _build_pia_attacker(
        components_module=components_module,
        model=model,
        attacker_name=plan.attacker_name,
        attack_num=plan.attack_num,
        interval=plan.interval,
        device=device,
    )

    with np.load(asset_summary["paths"]["member_split"]) as split_payload:
        member_all = set(int(item) for item in split_payload["mia_train_idxs"].tolist())
        nonmember_all = set(int(item) for item in split_payload["mia_eval_idxs"].tolist())
    if int(member_index) not in member_all:
        raise ValueError(f"Requested member_index is not in PIA member split: {member_index}")
    if int(nonmember_index) not in nonmember_all:
        raise ValueError(f"Requested nonmember_index is not in PIA non-member split: {nonmember_index}")

    member_indices = [int(member_index)]
    nonmember_indices = [int(nonmember_index)]
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
    member_batch = _collect_loader_batches(member_loader)
    nonmember_batch = _collect_loader_batches(nonmember_loader)
    if member_batch.shape[0] != 1 or nonmember_batch.shape[0] != 1:
        raise ValueError("PIA translated alias probe currently requires exactly one member and one non-member sample")

    baseline_member_forward = _run_pia_alias_forward(
        model=model,
        alias_selector=alias_selector,
        batch=member_batch,
        channel_dim=channel_dim,
        timestep=alias_timestep,
        channel_indices=None,
        alpha=1.0,
    )
    baseline_nonmember_forward = _run_pia_alias_forward(
        model=model,
        alias_selector=alias_selector,
        batch=nonmember_batch,
        channel_dim=channel_dim,
        timestep=alias_timestep,
        channel_indices=None,
        alpha=1.0,
    )
    baseline_profiles = {
        "member": _channelwise_profile_for_dim(baseline_member_forward["post_activation"], channel_dim=channel_dim),
        "nonmember": _channelwise_profile_for_dim(baseline_nonmember_forward["post_activation"], channel_dim=channel_dim),
    }
    channel_indices = _select_mask_indices(
        canary_profile=baseline_profiles["member"],
        control_profile=baseline_profiles["nonmember"],
        mask_kind=mask_kind,
        k=k,
        random_seed=mask_seed,
    )
    selected_index_tensor = torch.tensor(channel_indices, dtype=torch.long)
    all_indices = torch.arange(int(baseline_profiles["member"].shape[0]), dtype=torch.long)
    off_mask_indices = all_indices[~torch.isin(all_indices, selected_index_tensor)].tolist()

    intervened_member_forward = _run_pia_alias_forward(
        model=model,
        alias_selector=alias_selector,
        batch=member_batch,
        channel_dim=channel_dim,
        timestep=alias_timestep,
        channel_indices=channel_indices,
        alpha=alpha,
    )
    intervened_nonmember_forward = _run_pia_alias_forward(
        model=model,
        alias_selector=alias_selector,
        batch=nonmember_batch,
        channel_dim=channel_dim,
        timestep=alias_timestep,
        channel_indices=channel_indices,
        alpha=alpha,
    )
    intervened_profiles = {
        "member": _channelwise_profile_for_dim(intervened_member_forward["post_activation"], channel_dim=channel_dim),
        "nonmember": _channelwise_profile_for_dim(
            intervened_nonmember_forward["post_activation"],
            channel_dim=channel_dim,
        ),
    }

    pre_selected_delta = float(
        torch.abs(
            baseline_profiles["member"][selected_index_tensor] - baseline_profiles["nonmember"][selected_index_tensor]
        ).mean().item()
    )
    post_selected_delta = float(
        torch.abs(
            intervened_profiles["member"][selected_index_tensor] - intervened_profiles["nonmember"][selected_index_tensor]
        ).mean().item()
    )
    selected_delta_retention_ratio = float(post_selected_delta / pre_selected_delta) if pre_selected_delta else 0.0
    off_mask_drift_values: list[float] = []
    if off_mask_indices:
        off_mask_index_tensor = torch.tensor(off_mask_indices, dtype=torch.long)
        for role in ("member", "nonmember"):
            drift = torch.abs(
                intervened_profiles[role][off_mask_index_tensor] - baseline_profiles[role][off_mask_index_tensor]
            ).mean()
            off_mask_drift_values.append(float(drift.item()))
    off_mask_drift = float(sum(off_mask_drift_values) / len(off_mask_drift_values)) if off_mask_drift_values else 0.0

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
    baseline_member_scores, baseline_member_adaptive, baseline_member_std, baseline_member_probe = (
        _score_loader_with_alias_probe(
            attacker,
            model=model,
            loader=member_loader,
            alias_selector=alias_selector,
            channel_dim=channel_dim,
            device=device,
            adaptive_query_repeats=adaptive_query_repeats,
            channel_indices=None,
            alpha=1.0,
        )
    )
    baseline_nonmember_scores, baseline_nonmember_adaptive, baseline_nonmember_std, baseline_nonmember_probe = (
        _score_loader_with_alias_probe(
            attacker,
            model=model,
            loader=nonmember_loader,
            alias_selector=alias_selector,
            channel_dim=channel_dim,
            device=device,
            adaptive_query_repeats=adaptive_query_repeats,
            channel_indices=None,
            alpha=1.0,
        )
    )
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
    intervened_member_scores, intervened_member_adaptive, intervened_member_std, intervened_member_probe = (
        _score_loader_with_alias_probe(
            attacker,
            model=model,
            loader=member_loader,
            alias_selector=alias_selector,
            channel_dim=channel_dim,
            device=device,
            adaptive_query_repeats=adaptive_query_repeats,
            channel_indices=channel_indices,
            alpha=alpha,
        )
    )
    intervened_nonmember_scores, intervened_nonmember_adaptive, intervened_nonmember_std, intervened_nonmember_probe = (
        _score_loader_with_alias_probe(
            attacker,
            model=model,
            loader=nonmember_loader,
            alias_selector=alias_selector,
            channel_dim=channel_dim,
            device=device,
            adaptive_query_repeats=adaptive_query_repeats,
            channel_indices=channel_indices,
            alpha=alpha,
        )
    )

    member_score_mean = float(baseline_member_scores.mean().item())
    nonmember_score_mean = float(baseline_nonmember_scores.mean().item())
    intervened_member_score_mean = float(intervened_member_scores.mean().item())
    intervened_nonmember_score_mean = float(intervened_nonmember_scores.mean().item())
    baseline_gap = float(member_score_mean - nonmember_score_mean)
    intervened_gap = float(intervened_member_score_mean - intervened_nonmember_score_mean)

    score_records: list[dict[str, Any]] = []
    sample_specs = (
        (
            "member",
            member_indices[0],
            baseline_member_scores,
            baseline_member_adaptive,
            baseline_member_std,
            intervened_member_scores,
            intervened_member_adaptive,
            intervened_member_std,
            baseline_member_forward,
            intervened_member_forward,
            baseline_member_probe,
            intervened_member_probe,
        ),
        (
            "nonmember",
            nonmember_indices[0],
            baseline_nonmember_scores,
            baseline_nonmember_adaptive,
            baseline_nonmember_std,
            intervened_nonmember_scores,
            intervened_nonmember_adaptive,
            intervened_nonmember_std,
            baseline_nonmember_forward,
            intervened_nonmember_forward,
            baseline_nonmember_probe,
            intervened_nonmember_probe,
        ),
    )

    for membership, split_index, baseline_score, baseline_adaptive, baseline_std, intervened_score, intervened_adaptive, intervened_std, baseline_forward, intervened_forward, baseline_probe, intervened_probe in sample_specs:
        sample_dir = tensors_root / f"{membership}-{int(split_index):05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        baseline_alias_path = sample_dir / f"{alias_selector.replace('.', '_')}_baseline_alias.pt"
        intervened_alias_path = sample_dir / f"{alias_selector.replace('.', '_')}_intervened_alias.pt"
        baseline_prediction_path = sample_dir / f"{alias_selector.replace('.', '_')}_baseline_prediction.pt"
        intervened_prediction_path = sample_dir / f"{alias_selector.replace('.', '_')}_intervened_prediction.pt"
        torch.save(baseline_forward["post_activation"], baseline_alias_path)
        torch.save(intervened_forward["post_activation"], intervened_alias_path)
        torch.save(baseline_forward["prediction"], baseline_prediction_path)
        torch.save(intervened_forward["prediction"], intervened_prediction_path)
        score_records.append(
            {
                "membership": membership,
                "packet_position": 0,
                "split_index": int(split_index),
                "canonical_index": int(split_index),
                "baseline_score": round(float(baseline_score[0].item()), 6),
                "intervened_score": round(float(intervened_score[0].item()), 6),
                "score_delta": round(float((intervened_score - baseline_score)[0].item()), 6),
                "baseline_adaptive_score": round(float(baseline_adaptive[0].item()), 6),
                "intervened_adaptive_score": round(float(intervened_adaptive[0].item()), 6),
                "adaptive_score_delta": round(float((intervened_adaptive - baseline_adaptive)[0].item()), 6),
                "baseline_score_std": round(float(baseline_std[0].item()), 6),
                "intervened_score_std": round(float(intervened_std[0].item()), 6),
                "alias_activation_shape": list(baseline_forward["post_activation"].shape),
                "hook_hits": {
                    "baseline_forward": int(baseline_forward["hook_hits"]),
                    "intervened_forward": int(intervened_forward["hook_hits"]),
                    "baseline_score_path": int(baseline_probe["hook_hits"]),
                    "intervened_score_path": int(intervened_probe["hook_hits"]),
                },
                "artifact_paths": {
                    "baseline_alias": baseline_alias_path.relative_to(workspace_path).as_posix(),
                    "intervened_alias": intervened_alias_path.relative_to(workspace_path).as_posix(),
                    "baseline_prediction": baseline_prediction_path.relative_to(workspace_path).as_posix(),
                    "intervened_prediction": intervened_prediction_path.relative_to(workspace_path).as_posix(),
                },
            }
        )

    records_path = workspace_path / "sample_scores.jsonl"
    records_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in score_records) + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema": "diffaudit.pia.translated_alias_probe.v1",
        "status": "ready",
        "track": "gray-box",
        "method": "pia",
        "paper": "PIA_ICLR2024",
        "mode": "translated-alias-probe",
        "device": "cpu",
        "gpu_release": "none",
        "admitted_change": "none",
        "provenance_status": provenance_status,
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
            "sample_scores": str(records_path),
            "tensors_root": str(tensors_root),
        },
        "runtime": {
            "repo_root": str(Path(repo_root)),
            "dataset_root": plan.data_root,
            "member_split_root": str(split_root),
            "weights_key": weights_key,
            "attack_num": plan.attack_num,
            "interval": plan.interval,
            "batch_size": int(batch_size),
            "packet_size": 1,
            "adaptive_query_repeats": int(max(adaptive_query_repeats, 1)),
            "alias_timestep": int(alias_timestep),
        },
        "translation": {
            "alias_selector": alias_selector,
            "translated_from": translated_from,
            "translation_kind": "translated-contract",
            "translation_not_same_spec": True,
            "same_spec_reuse": False,
            "channel_dim": int(channel_dim),
            "tensor_layout": "BCHW",
            "alias_weight_shape": baseline_member_forward["alias_weight_shape"],
            "alias_activation_shape": list(baseline_member_forward["post_activation"].shape),
            "selector_resolved": True,
            "hook_hits": {
                "baseline_score_path": int(
                    baseline_member_probe["hook_hits"] + baseline_nonmember_probe["hook_hits"]
                ),
                "intervened_score_path": int(
                    intervened_member_probe["hook_hits"] + intervened_nonmember_probe["hook_hits"]
                ),
                "baseline_forward": int(
                    baseline_member_forward["hook_hits"] + baseline_nonmember_forward["hook_hits"]
                ),
                "intervened_forward": int(
                    intervened_member_forward["hook_hits"] + intervened_nonmember_forward["hook_hits"]
                ),
            },
        },
        "mask": {
            "mask_kind": mask_kind,
            "channel_indices": channel_indices,
            "k": int(len(channel_indices)),
            "alpha": float(alpha),
        },
        "packet": {
            "membership_semantics": "pia_split_v1",
            "canonical_index_authority": "pia_split_indices",
            "member_indices": member_indices,
            "nonmember_indices": nonmember_indices,
            "member_canonical_indices": member_indices,
            "nonmember_canonical_indices": nonmember_indices,
            "packet_positions": {
                "member": [0],
                "nonmember": [0],
            },
            "baseline_member_score_mean": round(member_score_mean, 6),
            "baseline_nonmember_score_mean": round(nonmember_score_mean, 6),
            "baseline_member_control_score_gap": round(baseline_gap, 6),
            "intervened_member_score_mean": round(intervened_member_score_mean, 6),
            "intervened_nonmember_score_mean": round(intervened_nonmember_score_mean, 6),
            "intervened_member_control_score_gap": round(intervened_gap, 6),
            "member_control_score_gap_delta": round(intervened_gap - baseline_gap, 6),
        },
        "metrics": {
            "selected_channel_abs_delta_pre": round(pre_selected_delta, 6),
            "selected_channel_abs_delta_post": round(post_selected_delta, 6),
            "selected_delta_retention_ratio": round(selected_delta_retention_ratio, 6),
            "off_mask_drift": round(off_mask_drift, 6),
        },
        "checks": {
            **asset_summary["checks"],
            "sample_scores_written": len(score_records),
            "packet_scores_generated": True,
            "device_cpu_only": True,
            "selector_resolved": True,
            "translation_not_same_spec": True,
        },
        "notes": [
            "This probe is an alias-scoped translated-contract canary, not same-spec reuse.",
            "Gray-box channel masking is applied on BCHW activations with channel_dim=1 at middleblocks.0.attn.proj_v.",
            "Packet deltas are local execution signals only and do not count as cross-box support by themselves.",
        ],
    }
    (workspace_path / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary


def run_synthetic_pia_smoke(
    workspace: str | Path,
    repo_root: str | Path = "external/PIA",
    device: str = "cpu",
    attacker_name: str = "PIA",
    attack_num: int = 3,
    interval: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    smoke_assets = bootstrap_pia_smoke_assets(
        target_dir=workspace_path / "synthetic-pia-assets",
        repo_root=repo_root,
    )
    components_module, model_module = load_pia_ddpm_modules(repo_root)
    model, weights_key = load_pia_model(smoke_assets["checkpoint_path"], model_module, device=device)
    attacker = _build_pia_attacker(
        components_module=components_module,
        model=model,
        attacker_name=attacker_name,
        attack_num=attack_num,
        interval=interval,
        device=device,
    )

    member_batch = torch.rand(4, 3, 32, 32, device=device)
    nonmember_batch = torch.clamp(torch.rand(4, 3, 32, 32, device=device) * 0.7 + 0.15, 0.0, 1.0)
    member_distances = attacker(member_batch).mean(dim=0)
    nonmember_distances = attacker(nonmember_batch).mean(dim=0)
    member_scores = -member_distances
    nonmember_scores = -nonmember_distances
    auc = _compute_auc(member_scores, nonmember_scores)
    threshold = torch.cat([member_scores, nonmember_scores]).median()
    asr = float(
        (
            torch.cat([member_scores > threshold, nonmember_scores <= threshold]).float().mean().item()
        )
    )

    result = {
        "status": "ready",
        "track": "gray-box",
        "method": "pia",
        "paper": "PIA_ICLR2024",
        "mode": "synthetic-smoke",
        "device": device,
        "workspace": str(workspace_path),
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
        },
        "assets": {
            "repo_root": str(Path(repo_root)),
            "weights_key": weights_key,
            "synthetic_checkpoint_path": smoke_assets["checkpoint_path"],
        },
        "checks": {
            "components_loaded": True,
            "model_loaded": True,
            "attacker_instantiated": True,
            "member_forward": True,
            "nonmember_forward": True,
            "synthetic_assets_cleaned": True,
        },
        "metrics": {
            "auc": round(float(auc), 6),
            "asr": round(float(asr), 6),
            "member_score_mean": round(float(member_scores.mean().item()), 6),
            "nonmember_score_mean": round(float(nonmember_scores.mean().item()), 6),
        },
        "cost": {
            "queries_per_sample": attack_num + 1,
            "timesteps": attack_num,
            "interval": interval,
            "batch_size": 4,
        },
        "notes": [
            "Synthetic smoke only verifies the PIA DDPM runtime path.",
            "Metrics are not benchmark numbers and should not be compared to paper results.",
        ],
    }
    (workspace_path / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    shutil.rmtree(workspace_path / "synthetic-pia-assets", ignore_errors=True)
    return result


def export_pia_packet_scores(
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
    adaptive_query_repeats: int = 1,
    provenance_status: str = "workspace-verified",
) -> dict[str, Any]:
    if device.lower() != "cpu":
        raise ValueError("PIA packet export is CPU-first and does not authorize GPU use.")

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
            "method": "pia",
            "paper": "PIA_ICLR2024",
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

    components_module, model_module = load_pia_ddpm_modules(repo_root)
    model, weights_key = load_pia_model(asset_summary["paths"]["checkpoint"], model_module, device=device)
    attacker = _build_pia_attacker(
        components_module=components_module,
        model=model,
        attacker_name=plan.attacker_name,
        attack_num=plan.attack_num,
        interval=plan.interval,
        device=device,
    )

    split_payload = np.load(asset_summary["paths"]["member_split"])
    member_all = split_payload["mia_train_idxs"].tolist()
    nonmember_all = split_payload["mia_eval_idxs"].tolist()
    if (member_index_file is None) != (nonmember_index_file is None):
        raise ValueError("PIA explicit packet export requires both member_index_file and nonmember_index_file")

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
            raise ValueError("Requested PIA packet exceeds available member/non-member indices")

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

    member_scores, member_adaptive_scores, member_score_std = _score_loader(
        attacker,
        member_loader,
        device=device,
        adaptive_query_repeats=adaptive_query_repeats,
    )
    nonmember_scores, nonmember_adaptive_scores, nonmember_score_std = _score_loader(
        attacker,
        nonmember_loader,
        device=device,
        adaptive_query_repeats=adaptive_query_repeats,
    )

    records: list[dict[str, Any]] = []
    for position, (split_index, score, adaptive_score, score_std) in enumerate(
        zip(member_indices, member_scores.tolist(), member_adaptive_scores.tolist(), member_score_std.tolist(), strict=True)
    ):
        records.append(
            {
                "membership": "member",
                "packet_position": int(position),
                "split_index": int(split_index),
                "score": round(float(score), 6),
                "adaptive_score": round(float(adaptive_score), 6),
                "score_std": round(float(score_std), 6),
            }
        )
    for position, (split_index, score, adaptive_score, score_std) in enumerate(
        zip(nonmember_indices, nonmember_scores.tolist(), nonmember_adaptive_scores.tolist(), nonmember_score_std.tolist(), strict=True)
    ):
        records.append(
            {
                "membership": "nonmember",
                "packet_position": int(position),
                "split_index": int(split_index),
                "score": round(float(score), 6),
                "adaptive_score": round(float(adaptive_score), 6),
                "score_std": round(float(score_std), 6),
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
        "method": "pia",
        "paper": "PIA_ICLR2024",
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
            "attack_num": plan.attack_num,
            "interval": plan.interval,
            "batch_size": int(batch_size),
            "selection_mode": selection_mode,
            "packet_size": int(selected_packet_size) if selected_packet_size is not None else None,
            "member_packet_size": int(len(member_indices)),
            "nonmember_packet_size": int(len(nonmember_indices)),
            "adaptive_query_repeats": int(max(adaptive_query_repeats, 1)),
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
            "This scaffold exports packet-local PIA scores on a fixed member/non-member packet.",
            "Exact-index mode can export pairboard-ready CPU-first scores on explicit PIA split indices.",
            "It is a CPU-first bridge artifact, not a benchmark or GPU release.",
        ],
    }
    (workspace_path / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return result


def run_pia_runtime_smoke(
    workspace: str | Path,
    repo_root: str | Path = "external/PIA",
    device: str = "cpu",
    attacker_name: str = "PIA",
    attack_num: int = 3,
    interval: int = 10,
) -> dict[str, Any]:
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    synthetic_root = workspace_path / "synthetic-assets"
    dataset_root = synthetic_root / "datasets"
    model_dir = synthetic_root / "model"
    member_split_root = synthetic_root / "member_splits"
    (dataset_root / "cifar10").mkdir(parents=True, exist_ok=True)
    member_split_root.mkdir(parents=True, exist_ok=True)
    (member_split_root / "CIFAR10_train_ratio0.5.npz").write_bytes(b"split")
    bootstrap_pia_smoke_assets(model_dir, repo_root)

    config = AuditConfig(
        task=TaskConfig(
            name="pia-runtime-smoke",
            model_family="diffusion",
            access_level="semi_white_box",
        ),
        assets=AssetConfig(
            dataset_id="synthetic-cifar10-half",
            model_id="synthetic-pia-ddpm",
            dataset_name="cifar10",
            dataset_root=dataset_root.as_posix(),
            model_dir=model_dir.as_posix(),
        ),
        attack=AttackConfig(
            method="pia",
            num_samples=8,
            parameters={
                "attacker_name": attacker_name,
                "attack_num": attack_num,
                "interval": interval,
            },
        ),
        report=ReportConfig(
            output_dir="experiments/pia-runtime-smoke"
        ),
    )
    exit_code, payload = probe_pia_runtime(
        config,
        repo_root=repo_root,
        member_split_root=member_split_root,
        device=device,
    )

    result = {
        "status": payload["status"],
        "track": "gray-box",
        "method": "pia",
        "paper": "PIA_ICLR2024",
        "mode": "runtime-smoke",
        "device": device,
        "workspace": str(workspace_path),
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
        },
        "assets": {
            "repo_root": str(Path(repo_root)),
            "dataset_name": "cifar10",
            "attacker_name": attacker_name,
            "attack_num": attack_num,
            "interval": interval,
        },
        "checks": {
            **payload.get("checks", {}),
            "runtime_probe_ready": exit_code == 0 and payload["status"] == "ready",
            "synthetic_assets_cleaned": True,
        },
        "notes": [
            "Runtime smoke uses synthetic assets to verify config-driven PIA readiness.",
            "Synthetic assets are removed after the summary is written.",
        ],
    }
    if "preview_score_shape" in payload:
        result["preview_score_shape"] = payload["preview_score_shape"]
    if "weights_key" in payload:
        result["weights_key"] = payload["weights_key"]
    if exit_code != 0 and "error" in payload:
        result["error"] = payload["error"]

    (workspace_path / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    shutil.rmtree(synthetic_root, ignore_errors=True)
    return result


def run_pia_runtime_mainline(
    config: AuditConfig,
    workspace: str | Path,
    repo_root: str | Path = "external/PIA",
    member_split_root: str | Path | None = None,
    device: str = "cpu",
    max_samples: int | None = None,
    batch_size: int = 8,
    stochastic_dropout_defense: bool = False,
    dropout_activation_schedule: str = "off",
    epsilon_output_noise_std: float = 0.0,
    epsilon_precision_bins: int | None = None,
    input_gaussian_blur_sigma: float = 0.0,
    adaptive_query_repeats: int = 1,
    late_step_threshold: int | None = None,
    provenance_status: str = "source-retained-unverified",
) -> dict[str, Any]:
    started_at = time.perf_counter()
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    resolved_schedule = _normalize_dropout_activation_schedule(
        dropout_activation_schedule,
        stochastic_dropout_defense=stochastic_dropout_defense,
    )

    exit_code, readiness = probe_pia_runtime(
        config,
        repo_root=repo_root,
        member_split_root=member_split_root,
        device=device,
    )
    if exit_code != 0 or readiness["status"] != "ready":
        result = {
            "status": "blocked",
            "track": "gray-box",
            "method": "pia",
            "paper": "PIA_ICLR2024",
            "mode": "runtime-mainline",
            "workspace": str(workspace_path),
            "workspace_name": workspace_path.name,
            "contract_stage": "target",
            "asset_grade": "single-machine-real-asset",
            "provenance_status": provenance_status,
            "evidence_level": "runtime-mainline",
            "checks": {
                "runtime_probe_ready": False,
            },
            "runtime_probe": readiness,
            "artifact_paths": {
                "summary": str(workspace_path / "summary.json"),
            },
            "notes": [
                "Runtime mainline requires the canonical checkpoint, dataset root, and member split to be ready.",
            ],
        }
        (workspace_path / "summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return result

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
    late_step_threshold = _resolve_late_step_threshold(
        attack_num=runtime_context.plan.attack_num,
        interval=runtime_context.plan.interval,
        late_step_threshold=late_step_threshold,
    )
    model.eval()
    attacker = _build_pia_attacker(
        components_module=runtime_context.components_module,
        model=model,
        attacker_name=runtime_context.plan.attacker_name,
        attack_num=runtime_context.plan.attack_num,
        interval=runtime_context.plan.interval,
        device=device,
        dropout_activation_schedule=resolved_schedule,
        late_step_threshold=late_step_threshold,
        epsilon_output_noise_std=epsilon_output_noise_std,
        epsilon_precision_bins=epsilon_precision_bins,
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

    member_score_tensor, member_adaptive_tensor, member_score_std_tensor = _score_loader(
        attacker,
        member_loader,
        device=device,
        adaptive_query_repeats=adaptive_query_repeats,
        input_gaussian_blur_sigma=input_gaussian_blur_sigma,
    )
    nonmember_score_tensor, nonmember_adaptive_tensor, nonmember_score_std_tensor = _score_loader(
        attacker,
        nonmember_loader,
        device=device,
        adaptive_query_repeats=adaptive_query_repeats,
        input_gaussian_blur_sigma=input_gaussian_blur_sigma,
    )
    metrics = {
        "auc": round(_compute_auc(member_score_tensor, nonmember_score_tensor), 6),
        **_compute_threshold_metrics(
            member_score_tensor.numpy(),
            nonmember_score_tensor.numpy(),
        ),
        "member_score_mean": round(float(member_score_tensor.mean().item()), 6),
        "nonmember_score_mean": round(float(nonmember_score_tensor.mean().item()), 6),
    }
    adaptive_check = {
        "status": "completed" if adaptive_query_repeats > 1 else "not_requested",
        "enabled": bool(adaptive_query_repeats > 1),
        "query_repeats": int(max(adaptive_query_repeats, 1)),
        "aggregation": "mean",
        "metrics": {
            "auc": round(_compute_auc(member_adaptive_tensor, nonmember_adaptive_tensor), 6),
            **_compute_threshold_metrics(
                member_adaptive_tensor.numpy(),
                nonmember_adaptive_tensor.numpy(),
            ),
            "member_score_mean": round(float(member_adaptive_tensor.mean().item()), 6),
            "nonmember_score_mean": round(float(nonmember_adaptive_tensor.mean().item()), 6),
        },
        "score_std": {
            "member_mean": round(float(member_score_std_tensor.mean().item()), 6),
            "nonmember_mean": round(float(nonmember_score_std_tensor.mean().item()), 6),
        },
    }
    scores_path = workspace_path / "scores.json"
    adaptive_scores_path = workspace_path / "adaptive-scores.json"
    scores_payload = {
        "member_scores": [round(float(value), 6) for value in member_score_tensor.tolist()],
        "nonmember_scores": [round(float(value), 6) for value in nonmember_score_tensor.tolist()],
        "member_indices": member_indices,
        "nonmember_indices": nonmember_indices,
    }
    scores_path.write_text(
        json.dumps(scores_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    adaptive_scores_payload = {
        "query_repeats": int(max(adaptive_query_repeats, 1)),
        "aggregation": "mean",
        "member_scores": [round(float(value), 6) for value in member_adaptive_tensor.tolist()],
        "nonmember_scores": [round(float(value), 6) for value in nonmember_adaptive_tensor.tolist()],
        "member_score_std": [round(float(value), 6) for value in member_score_std_tensor.tolist()],
        "nonmember_score_std": [round(float(value), 6) for value in nonmember_score_std_tensor.tolist()],
        "member_indices": member_indices,
        "nonmember_indices": nonmember_indices,
    }
    adaptive_scores_path.write_text(
        json.dumps(adaptive_scores_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    member_batches = _collect_loader_batches(member_loader)
    nonmember_batches = _collect_loader_batches(nonmember_loader)
    reference_batches = torch.cat([member_batches, nonmember_batches], dim=0)
    baseline_surrogates = _predict_surrogate_images(
        model,
        reference_batches,
        device=device,
        dropout_activation_schedule="off",
        late_step_threshold=late_step_threshold,
        epsilon_output_noise_std=0.0,
        epsilon_precision_bins=None,
        input_gaussian_blur_sigma=0.0,
    )
    active_surrogates = _predict_surrogate_images(
        model,
        reference_batches,
        device=device,
        dropout_activation_schedule=resolved_schedule,
        late_step_threshold=late_step_threshold,
        epsilon_output_noise_std=epsilon_output_noise_std,
        epsilon_precision_bins=epsilon_precision_bins,
        input_gaussian_blur_sigma=input_gaussian_blur_sigma,
    )
    quality = _compute_surrogate_quality_metrics(
        reference_images=reference_batches,
        active_images=active_surrogates,
        baseline_images=baseline_surrogates if (resolved_schedule != "off" or float(epsilon_output_noise_std) > 0.0 or epsilon_precision_bins is not None or float(input_gaussian_blur_sigma) > 0.0) else None,
    )

    defense_name = "none"
    defense_stage = "baseline"
    if epsilon_precision_bins is not None:
        defense_name = "epsilon-precision-throttling"
        defense_stage = "candidate-g2"
    elif stochastic_dropout_defense and float(epsilon_output_noise_std) > 0.0:
        defense_name = "stochastic-dropout+epsilon-output-noise"
        defense_stage = "candidate-g2"
    elif float(input_gaussian_blur_sigma) > 0.0 and stochastic_dropout_defense:
        defense_name = "stochastic-dropout+input-gaussian-blur"
        defense_stage = "candidate-g2"
    elif float(input_gaussian_blur_sigma) > 0.0:
        defense_name = "input-gaussian-blur"
        defense_stage = "candidate-g2"
    elif stochastic_dropout_defense:
        defense_name = "stochastic-dropout"
        defense_stage = "provisional-g1"
    elif float(epsilon_output_noise_std) > 0.0:
        defense_name = "epsilon-output-noise"
        defense_stage = "candidate-g2"

    result = {
        "status": "ready",
        "track": "gray-box",
        "method": "pia",
        "paper": "PIA_ICLR2024",
        "mode": "runtime-mainline",
        "device": device,
        "workspace": str(workspace_path),
        "workspace_name": workspace_path.name,
        "contract_stage": "target",
        "asset_grade": "single-machine-real-asset",
        "provenance_status": provenance_status,
        "evidence_level": "runtime-mainline",
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
            "scores": str(scores_path),
            "adaptive_scores": str(adaptive_scores_path),
        },
        "checks": {
            **asset_summary["checks"],
            "runtime_probe_ready": True,
            "member_scores_generated": True,
            "nonmember_scores_generated": True,
            "adaptive_scores_generated": True,
        },
        "runtime": {
            "repo_root": str(Path(repo_root)),
            "model_dir": runtime_context.plan.model_dir,
            "dataset_root": runtime_context.plan.data_root,
            "member_split_root": str(split_root),
            "batch_size": int(batch_size),
            "max_samples": int(selected_max_samples),
            "num_samples": int(member_score_tensor.shape[0]),
            "attack_num": runtime_context.plan.attack_num,
            "interval": runtime_context.plan.interval,
            "weights_key": weights_key,
            "elapsed_seconds": round(time.perf_counter() - started_at, 6),
        },
        "defense": {
            "name": defense_name,
            "enabled": bool(stochastic_dropout_defense or float(epsilon_output_noise_std) > 0.0 or epsilon_precision_bins is not None or float(input_gaussian_blur_sigma) > 0.0),
            "dropout_activation_schedule": resolved_schedule,
            "late_step_threshold": int(late_step_threshold),
            "epsilon_output_noise_std": float(epsilon_output_noise_std),
            "epsilon_precision_bins": int(epsilon_precision_bins) if epsilon_precision_bins is not None else None,
            "input_gaussian_blur_sigma": float(input_gaussian_blur_sigma),
        },
        "defense_stage": defense_stage,
        "sample_count_per_split": int(member_score_tensor.shape[0]),
        "metrics": metrics,
        "adaptive_check": adaptive_check,
        "quality": quality,
        "cost": {
            "device": device,
            "execution_mode": "single-device-serial",
            "sample_count_per_split": int(member_score_tensor.shape[0]),
            "batch_size": int(batch_size),
            "attack_num": int(runtime_context.plan.attack_num),
            "interval": int(runtime_context.plan.interval),
            "queries_per_sample": int(runtime_context.plan.attack_num),
            "model_queries_per_sample": int(runtime_context.plan.attack_num + 1),
            "adaptive_query_repeats": int(max(adaptive_query_repeats, 1)),
            "wall_clock_seconds": round(time.perf_counter() - started_at, 6),
        },
        "notes": [
            "Runtime mainline executes the current PIA DDPM path on canonical real assets.",
            "Current evidence is suitable for a local baseline and must not be overstated as benchmark-ready.",
        ],
    }
    (workspace_path / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return result
