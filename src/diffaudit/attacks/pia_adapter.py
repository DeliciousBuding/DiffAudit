"""Runtime helpers for the official PIA DDPM implementation."""

from __future__ import annotations

import importlib.util
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from diffaudit.attacks.pia import (
    PiaPlan,
    build_pia_plan,
    probe_pia_assets,
    validate_pia_workspace,
)
from diffaudit.config import AuditConfig


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
):
    class RuntimeEpsGetter(components_module.EpsGetter):
        def __call__(
            self,
            xt: torch.Tensor,
            condition: torch.Tensor | None = None,
            noise_level: torch.Tensor | None = None,
            t: int | None = None,
        ) -> torch.Tensor:
            del condition, noise_level
            timestep = torch.ones([xt.shape[0]], device=xt.device, dtype=torch.long) * int(t or 0)
            return self.model(xt, t=timestep)

    return RuntimeEpsGetter(model)


def _build_pia_attacker(
    components_module: ModuleType,
    model: torch.nn.Module,
    attacker_name: str,
    attack_num: int,
    interval: int,
    device: str,
    defaults: PiaDefaults | None = None,
):
    config = defaults or PiaDefaults()
    attacker_cls = getattr(components_module, attacker_name, None)
    if attacker_cls is None:
        raise ValueError(f"PIA attacker class not found: {attacker_name}")
    betas = torch.linspace(config.beta_1, config.beta_T, config.T, device=device)
    eps_getter = _build_pia_eps_getter(components_module, model)
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


def _compute_auc(member_scores: torch.Tensor, nonmember_scores: torch.Tensor) -> float:
    member = member_scores.detach().cpu().flatten()
    nonmember = nonmember_scores.detach().cpu().flatten()
    wins = (member[:, None] > nonmember[None, :]).float()
    ties = (member[:, None] == nonmember[None, :]).float() * 0.5
    return float((wins + ties).mean().item())


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
