"""Read-only helpers for the Finding NeMo migrated DDPM observability contract."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
from safetensors.torch import load_file

from diffaudit.attacks.gsa import _latest_checkpoint_dir, validate_gsa_workspace


SUPPORTED_SIGNAL_TYPES = {"activations", "grad_norm"}


def _build_gsa_unet(resolution: int = 32) -> UNet2DModel:
    return UNet2DModel(
        sample_size=resolution,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )


def _derive_class_hint(split_root: Path, sample_path: Path) -> str | None:
    relative_parent = sample_path.parent.relative_to(split_root)
    if str(relative_parent) != ".":
        return relative_parent.as_posix()
    stem = sample_path.stem
    if "-" in stem:
        return stem.split("-", 1)[0]
    return None


def _build_sample_index(split_root: str | Path) -> dict[str, dict[str, str | None]]:
    root = Path(split_root)
    index: dict[str, dict[str, str | None]] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        dataset_relpath = path.relative_to(root).as_posix()
        class_hint = _derive_class_hint(root, path)
        canonical_id = f"{root.name}/{dataset_relpath}"
        legacy_id = f"{root.name}:{path.stem}"
        bare_id = path.stem
        payload = {
            "sample_id": canonical_id,
            "dataset_relpath": dataset_relpath,
            "absolute_path": str(path.resolve()),
            "class_hint": class_hint,
            "binding_source": "filesystem-scan",
        }
        for key in (canonical_id, legacy_id, bare_id):
            index.setdefault(key, payload)
    return index


def resolve_gsa_sample_binding(
    assets_root: str | Path,
    split: str,
    sample_id: str,
) -> dict[str, str | None]:
    split_root = Path(assets_root) / "datasets" / split
    if not split_root.exists():
        raise FileNotFoundError(f"GSA split root not found: {split_root}")
    sample_index = _build_sample_index(split_root)
    if sample_id not in sample_index:
        raise FileNotFoundError(f"Sample id not found in split '{split}': {sample_id}")
    return sample_index[sample_id]


def resolve_gsa_layer_selector(
    layer_selector: str,
    resolution: int = 32,
) -> dict[str, Any]:
    model = _build_gsa_unet(resolution=resolution)
    modules = dict(model.named_modules())
    matches = [name for name in modules if name == layer_selector]
    if not matches:
        raise KeyError(f"Layer selector not found: {layer_selector}")
    if len(matches) > 1:
        raise ValueError(f"Layer selector resolved ambiguously: {layer_selector}")
    layer_id = matches[0]
    parameter_prefixes = sorted(
        key for key in model.state_dict().keys() if key.startswith(f"{layer_id}.")
    )
    if not parameter_prefixes:
        raise KeyError(f"No parameter prefixes found for selector: {layer_selector}")
    layer_family = layer_id.split(".", 1)[0]
    return {
        "layer_selector": layer_selector,
        "layer_id": layer_id,
        "layer_family": layer_family,
        "module_type": type(modules[layer_id]).__name__,
        "parameter_prefixes": parameter_prefixes,
    }


def probe_gsa_observability_contract(
    repo_root: str | Path,
    assets_root: str | Path,
    checkpoint_root: str | Path,
    split: str,
    sample_id: str,
    layer_selector: str,
    signal_type: str = "activations",
    resolution: int = 32,
    candidate: str = "Finding NeMo + local memorization + FB-Mem",
    provenance_status: str = "workspace-verified",
) -> dict[str, Any]:
    workspace = validate_gsa_workspace(repo_root)
    assets_path = Path(assets_root)
    checkpoint_path = Path(checkpoint_root)
    checks: dict[str, bool] = {
        "workspace_files": True,
        "assets_root_exists": assets_path.exists(),
        "checkpoint_root_exists": checkpoint_path.exists(),
        "signal_type_supported": signal_type in SUPPORTED_SIGNAL_TYPES,
    }
    missing: list[str] = []

    resolved_checkpoint_dir: Path | None = None
    if checks["checkpoint_root_exists"]:
        try:
            resolved_checkpoint_dir = _latest_checkpoint_dir(checkpoint_path)
            checks["resolved_checkpoint_exists"] = resolved_checkpoint_dir.exists()
        except FileNotFoundError:
            checks["resolved_checkpoint_exists"] = False
            missing.append(str(checkpoint_path))
    else:
        checks["resolved_checkpoint_exists"] = False
        missing.append(str(checkpoint_path))

    try:
        binding = resolve_gsa_sample_binding(assets_path, split=split, sample_id=sample_id)
        checks["sample_binding_resolved"] = True
    except FileNotFoundError as exc:
        binding = None
        checks["sample_binding_resolved"] = False
        missing.append(str(exc))

    try:
        selector = resolve_gsa_layer_selector(layer_selector, resolution=resolution)
        checks["layer_selector_resolved"] = True
    except (KeyError, ValueError) as exc:
        selector = None
        checks["layer_selector_resolved"] = False
        missing.append(str(exc))

    if not checks["signal_type_supported"]:
        missing.append(f"unsupported signal type: {signal_type}")

    status = "ready" if all(checks.values()) else "blocked"
    payload: dict[str, Any] = {
        "status": status,
        "track": "white-box",
        "method": "gsa-observability",
        "mode": "contract-probe",
        "candidate": candidate,
        "contract_stage": "stage-1-observability-smoke",
        "provenance_status": provenance_status,
        "gpu_release": "none",
        "admitted_change": "none",
        "workspace": workspace,
        "checks": checks,
        "requested": {
            "assets_root": str(assets_path),
            "checkpoint_root": str(checkpoint_path),
            "split": split,
            "sample_id": sample_id,
            "layer_selector": layer_selector,
            "signal_type": signal_type,
            "resolution": int(resolution),
        },
        "resolved": {
            "resolved_checkpoint_dir": str(resolved_checkpoint_dir) if resolved_checkpoint_dir else None,
            "sample_binding": binding,
            "layer_binding": selector,
        },
        "missing": missing,
        "notes": [
            "This probe is read-only and does not export activations.",
            "Ready means the contract fields resolve; it does not authorize any run.",
        ],
    }
    return payload


def _load_image_tensor(image_path: str | Path, resolution: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    if image.size != (resolution, resolution):
        image = image.resize((resolution, resolution))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor.mul(2.0).sub(1.0).unsqueeze(0)


def _sanitize_path_fragment(value: str) -> str:
    sanitized = value.replace("\\", "/")
    for token in ("/", ":", "."):
        sanitized = sanitized.replace(token, "_")
    return sanitized


def _resolve_checkpoint_dir(checkpoint_root: str | Path, checkpoint_dir: str | Path | None) -> Path:
    if checkpoint_dir is not None:
        path = Path(checkpoint_dir)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint dir not found: {path}")
        return path
    return _latest_checkpoint_dir(checkpoint_root)


def _load_gsa_unet_checkpoint(
    checkpoint_root: str | Path,
    checkpoint_dir: str | Path | None = None,
    resolution: int = 32,
    device: str = "cpu",
) -> tuple[UNet2DModel, Path]:
    resolved_checkpoint_dir = _resolve_checkpoint_dir(checkpoint_root, checkpoint_dir)
    weights_path = resolved_checkpoint_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"GSA checkpoint weights not found: {weights_path}")
    model = _build_gsa_unet(resolution=resolution)
    state_dict = load_file(str(weights_path))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, resolved_checkpoint_dir


def _build_gsa_noise_scheduler(
    ddpm_num_steps: int = 1000,
    prediction_type: str = "epsilon",
) -> DDPMScheduler:
    scheduler_kwargs: dict[str, Any] = {
        "num_train_timesteps": int(ddpm_num_steps),
        "beta_schedule": "linear",
    }
    if "prediction_type" in inspect.signature(DDPMScheduler.__init__).parameters:
        scheduler_kwargs["prediction_type"] = prediction_type
    return DDPMScheduler(**scheduler_kwargs)


def _prepare_noisy_sample(
    sample_tensor: torch.Tensor,
    timestep: int,
    noise_seed: int,
    prediction_type: str,
    ddpm_num_steps: int = 1000,
) -> torch.Tensor:
    scheduler = _build_gsa_noise_scheduler(
        ddpm_num_steps=ddpm_num_steps,
        prediction_type=prediction_type,
    )
    generator = torch.Generator(device=str(sample_tensor.device))
    generator.manual_seed(int(noise_seed))
    noise = torch.randn(
        sample_tensor.shape,
        generator=generator,
        device=sample_tensor.device,
        dtype=sample_tensor.dtype,
    )
    timestep_tensor = torch.tensor([int(timestep)], device=sample_tensor.device).long()
    return scheduler.add_noise(sample_tensor, noise, timestep_tensor)


def _capture_activation_tensor(
    model: UNet2DModel,
    layer_id: str,
    noisy_sample: torch.Tensor,
    timestep: int,
) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}
    modules = dict(model.named_modules())
    if layer_id not in modules:
        raise KeyError(f"Layer id not found in model: {layer_id}")

    def _hook(_: torch.nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if isinstance(output, tuple):
            tensor = output[0]
        else:
            tensor = output
        captured["activation"] = tensor.detach().cpu()

    handle = modules[layer_id].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(noisy_sample, timestep=torch.tensor([timestep], device=noisy_sample.device))
    finally:
        handle.remove()

    if "activation" not in captured:
        raise RuntimeError(f"Activation hook did not capture output for layer: {layer_id}")
    return captured["activation"]


def _extract_model_prediction(model_output: Any) -> torch.Tensor:
    if hasattr(model_output, "sample"):
        prediction = model_output.sample
    elif isinstance(model_output, tuple):
        prediction = model_output[0]
    else:
        prediction = model_output
    if not isinstance(prediction, torch.Tensor):
        raise TypeError(f"Unexpected model prediction type: {type(prediction).__name__}")
    return prediction


def _run_layer_intervened_forward(
    model: UNet2DModel,
    layer_id: str,
    noisy_sample: torch.Tensor,
    timestep: int,
    channel_indices: list[int] | None = None,
    alpha: float = 1.0,
) -> dict[str, Any]:
    captured: dict[str, torch.Tensor] = {}
    modules = dict(model.named_modules())
    if layer_id not in modules:
        raise KeyError(f"Layer id not found in model: {layer_id}")

    def _hook(_: torch.nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor):
        if isinstance(output, tuple):
            tensor = output[0]
        else:
            tensor = output
        captured["pre_activation"] = tensor.detach().cpu()
        if channel_indices:
            masked = _apply_channel_mask(tensor, channel_indices=channel_indices, alpha=alpha)
            captured["post_activation"] = masked.detach().cpu()
            captured["intervention_applied"] = torch.tensor(True)
            if isinstance(output, tuple):
                return (masked, *output[1:])
            return masked
        captured["post_activation"] = tensor.detach().cpu()
        captured["intervention_applied"] = torch.tensor(False)
        return output

    handle = modules[layer_id].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model_output = model(noisy_sample, timestep=torch.tensor([timestep], device=noisy_sample.device))
    finally:
        handle.remove()

    if "pre_activation" not in captured or "post_activation" not in captured:
        raise RuntimeError(f"Intervention hook did not capture output for layer: {layer_id}")
    prediction = _extract_model_prediction(model_output).detach().cpu()
    return {
        "pre_activation": captured["pre_activation"],
        "post_activation": captured["post_activation"],
        "prediction": prediction,
        "intervention_applied": bool(captured["intervention_applied"].item()),
    }


def _channelwise_profile(activation: torch.Tensor) -> torch.Tensor:
    if activation.ndim == 0:
        return activation.reshape(1)
    channel_dim = int(activation.shape[-1])
    return activation.reshape(-1, channel_dim).mean(dim=0)


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


def _apply_channel_mask(
    activation: torch.Tensor,
    channel_indices: list[int],
    alpha: float,
) -> torch.Tensor:
    masked = activation.clone()
    if channel_indices:
        masked[..., channel_indices] *= float(alpha)
    return masked


def export_gsa_observability_canary(
    workspace: str | Path,
    repo_root: str | Path,
    assets_root: str | Path,
    checkpoint_root: str | Path,
    checkpoint_dir: str | Path | None,
    split: str,
    sample_id: str,
    control_split: str,
    control_sample_id: str,
    layer_selector: str,
    signal_type: str = "activations",
    timestep: int = 999,
    noise_seed: int = 0,
    prediction_type: str = "epsilon",
    device: str = "cpu",
    resolution: int = 32,
    ddpm_num_steps: int = 1000,
    provenance_status: str = "workspace-verified",
    candidate: str = "Finding NeMo + local memorization + FB-Mem",
) -> dict[str, Any]:
    if device.lower() != "cpu":
        raise ValueError("Activation export canary is CPU-only and does not authorize GPU use.")
    if signal_type != "activations":
        raise ValueError("Activation export canary currently supports activations only.")

    workspace_info = validate_gsa_workspace(repo_root)
    assets_path = Path(assets_root)
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    records_path = workspace_path / "records.jsonl"
    tensors_root = workspace_path / "tensors"
    tensors_root.mkdir(parents=True, exist_ok=True)

    selector = resolve_gsa_layer_selector(layer_selector, resolution=resolution)
    model, resolved_checkpoint_dir = _load_gsa_unet_checkpoint(
        checkpoint_root=checkpoint_root,
        checkpoint_dir=checkpoint_dir,
        resolution=resolution,
        device="cpu",
    )

    requested = {
        "workspace": str(workspace_path),
        "assets_root": str(assets_path),
        "checkpoint_root": str(Path(checkpoint_root)),
        "checkpoint_dir": str(Path(checkpoint_dir)) if checkpoint_dir is not None else None,
        "split": split,
        "sample_id": sample_id,
        "control_split": control_split,
        "control_sample_id": control_sample_id,
        "layer_selector": layer_selector,
        "signal_type": signal_type,
        "timestep": int(timestep),
        "noise_seed": int(noise_seed),
        "prediction_type": prediction_type,
        "ddpm_num_steps": int(ddpm_num_steps),
        "device": "cpu",
        "resolution": int(resolution),
    }

    torch.manual_seed(int(noise_seed))
    sample_specs = (
        {"split": split, "sample_id": sample_id, "role": "canary"},
        {"split": control_split, "sample_id": control_sample_id, "role": "control"},
    )

    records: list[dict[str, Any]] = []
    artifact_paths: list[Path] = []
    for spec in sample_specs:
        binding = resolve_gsa_sample_binding(assets_path, split=spec["split"], sample_id=spec["sample_id"])
        sample_tensor = _load_image_tensor(binding["absolute_path"], resolution=resolution)
        noisy_sample = _prepare_noisy_sample(
            sample_tensor=sample_tensor,
            timestep=int(timestep),
            noise_seed=int(noise_seed),
            prediction_type=prediction_type,
            ddpm_num_steps=int(ddpm_num_steps),
        )
        activation = _capture_activation_tensor(
            model=model,
            layer_id=selector["layer_id"],
            noisy_sample=noisy_sample,
            timestep=int(timestep),
        )

        sample_dir = tensors_root / _sanitize_path_fragment(binding["sample_id"])
        sample_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = sample_dir / f"{_sanitize_path_fragment(layer_selector)}_t{int(timestep)}.pt"
        torch.save(activation, artifact_path)
        relative_artifact_path = artifact_path.relative_to(workspace_path)
        artifact_paths.append(artifact_path)

        summary_stat = {
            "mean": float(activation.mean().item()),
            "std": float(activation.std(unbiased=False).item()),
            "min": float(activation.min().item()),
            "max": float(activation.max().item()),
        }
        records.append(
            {
                "sample_id": binding["sample_id"],
                "split": spec["split"],
                "role": spec["role"],
                "dataset_relpath": binding["dataset_relpath"],
                "checkpoint_root": str(Path(checkpoint_root)),
                "resolved_checkpoint_dir": str(resolved_checkpoint_dir),
                "signal_type": signal_type,
                "layer_id": selector["layer_id"],
                "layer_selector": layer_selector,
                "timestep": int(timestep),
                "tensor_shape": list(activation.shape),
                "summary_stat": summary_stat,
                "artifact_path": relative_artifact_path.as_posix(),
            }
        )

    records_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema": "diffaudit.gsa_observability.canary.v1",
        "status": "ready",
        "track": "white-box",
        "method": "gsa-observability",
        "mode": "activation-export-canary",
        "candidate": candidate,
        "contract_stage": "stage-1-observability-smoke",
        "provenance_status": provenance_status,
        "gpu_release": "none",
        "admitted_change": "none",
        "workspace": workspace_info,
        "requested": requested,
        "checks": {
            "workspace_files": True,
            "records_written": len(records),
            "tensor_artifacts_written": len(artifact_paths),
            "sample_pair_resolved": len(records) == 2,
            "layer_selector_resolved": True,
            "device_cpu_only": True,
        },
        "artifact_paths": {
            "summary": "summary.json",
            "records": "records.jsonl",
            "tensors_root": "tensors",
        },
        "notes": [
            "This adapter exports sample-level activations only.",
            "The export path applies upstream-style DDPM noise before the hooked forward pass.",
            "Ready means the export path exists; it does not authorize any run or benchmark claim.",
        ],
    }
    (workspace_path / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary


def export_gsa_observability_masked_packet(
    workspace: str | Path,
    repo_root: str | Path,
    assets_root: str | Path,
    checkpoint_root: str | Path,
    checkpoint_dir: str | Path | None,
    split: str,
    sample_id: str,
    control_split: str,
    control_sample_id: str,
    layer_selector: str,
    mask_kind: str = "top_abs_delta_k",
    k: int = 8,
    alpha: float = 0.5,
    timestep: int = 999,
    noise_seed: int = 0,
    mask_seed: int = 0,
    prediction_type: str = "epsilon",
    device: str = "cpu",
    resolution: int = 32,
    ddpm_num_steps: int = 1000,
    provenance_status: str = "workspace-verified",
    candidate: str = "I-C white-gray bridge packet",
) -> dict[str, Any]:
    if device.lower() != "cpu":
        raise ValueError("Masked packet export is CPU-first and does not authorize GPU use.")

    workspace_info = validate_gsa_workspace(repo_root)
    assets_path = Path(assets_root)
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    records_path = workspace_path / "records.jsonl"
    tensors_root = workspace_path / "tensors"
    tensors_root.mkdir(parents=True, exist_ok=True)

    selector = resolve_gsa_layer_selector(layer_selector, resolution=resolution)
    model, resolved_checkpoint_dir = _load_gsa_unet_checkpoint(
        checkpoint_root=checkpoint_root,
        checkpoint_dir=checkpoint_dir,
        resolution=resolution,
        device="cpu",
    )

    sample_specs = (
        {"split": split, "sample_id": sample_id, "role": "canary"},
        {"split": control_split, "sample_id": control_sample_id, "role": "control"},
    )
    records: list[dict[str, Any]] = []
    activations_by_role: dict[str, torch.Tensor] = {}
    profiles_by_role: dict[str, torch.Tensor] = {}

    for spec in sample_specs:
        binding = resolve_gsa_sample_binding(assets_path, split=spec["split"], sample_id=spec["sample_id"])
        sample_tensor = _load_image_tensor(binding["absolute_path"], resolution=resolution)
        noisy_sample = _prepare_noisy_sample(
            sample_tensor=sample_tensor,
            timestep=int(timestep),
            noise_seed=int(noise_seed),
            prediction_type=prediction_type,
            ddpm_num_steps=int(ddpm_num_steps),
        )
        activation = _capture_activation_tensor(
            model=model,
            layer_id=selector["layer_id"],
            noisy_sample=noisy_sample,
            timestep=int(timestep),
        )
        activations_by_role[spec["role"]] = activation
        profiles_by_role[spec["role"]] = _channelwise_profile(activation)
        records.append(
            {
                "split": spec["split"],
                "sample_id": binding["sample_id"],
                "role": spec["role"],
                "dataset_relpath": binding["dataset_relpath"],
                "layer_id": selector["layer_id"],
                "layer_selector": layer_selector,
                "timestep": int(timestep),
                "tensor_shape": list(activation.shape),
            }
        )

    channel_indices = _select_mask_indices(
        canary_profile=profiles_by_role["canary"],
        control_profile=profiles_by_role["control"],
        mask_kind=mask_kind,
        k=k,
        random_seed=mask_seed,
    )
    selected_index_tensor = torch.tensor(channel_indices, dtype=torch.long)
    all_indices = torch.arange(int(profiles_by_role["canary"].shape[0]), dtype=torch.long)
    off_mask_indices = all_indices[~torch.isin(all_indices, selected_index_tensor)].tolist()

    masked_activations = {
        role: _apply_channel_mask(activation, channel_indices=channel_indices, alpha=alpha)
        for role, activation in activations_by_role.items()
    }
    masked_profiles = {role: _channelwise_profile(tensor) for role, tensor in masked_activations.items()}

    pre_selected_delta = float(
        torch.abs(
            profiles_by_role["canary"][selected_index_tensor] - profiles_by_role["control"][selected_index_tensor]
        ).mean().item()
    )
    post_selected_delta = float(
        torch.abs(
            masked_profiles["canary"][selected_index_tensor] - masked_profiles["control"][selected_index_tensor]
        ).mean().item()
    )
    selected_delta_retention_ratio = float(post_selected_delta / pre_selected_delta) if pre_selected_delta else 0.0

    off_mask_drift_values: list[float] = []
    if off_mask_indices:
        off_mask_index_tensor = torch.tensor(off_mask_indices, dtype=torch.long)
        for role in ("canary", "control"):
            drift = torch.abs(
                masked_profiles[role][off_mask_index_tensor] - profiles_by_role[role][off_mask_index_tensor]
            ).mean()
            off_mask_drift_values.append(float(drift.item()))
    off_mask_drift = float(sum(off_mask_drift_values) / len(off_mask_drift_values)) if off_mask_drift_values else 0.0

    artifact_paths: dict[str, dict[str, str]] = {}
    for record in records:
        role = str(record["role"])
        sample_dir = tensors_root / _sanitize_path_fragment(str(record["sample_id"]))
        sample_dir.mkdir(parents=True, exist_ok=True)
        pre_path = sample_dir / f"{_sanitize_path_fragment(layer_selector)}_t{int(timestep)}_pre.pt"
        post_path = sample_dir / f"{_sanitize_path_fragment(layer_selector)}_t{int(timestep)}_post.pt"
        torch.save(activations_by_role[role], pre_path)
        torch.save(masked_activations[role], post_path)
        artifact_paths[role] = {
            "pre_mask": pre_path.relative_to(workspace_path).as_posix(),
            "post_mask": post_path.relative_to(workspace_path).as_posix(),
        }
        record["artifact_paths"] = artifact_paths[role]

    records_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema": "diffaudit.gsa_observability.masked_packet.v1",
        "status": "ready",
        "track": "white-box",
        "method": "gsa-observability",
        "mode": "masked-packet-export",
        "candidate": candidate,
        "contract_stage": "stage-2-masked-packet-scaffold",
        "provenance_status": provenance_status,
        "gpu_release": "none",
        "admitted_change": "none",
        "workspace": workspace_info,
        "requested": {
            "workspace": str(workspace_path),
            "assets_root": str(assets_path),
            "checkpoint_root": str(Path(checkpoint_root)),
            "checkpoint_dir": str(Path(checkpoint_dir)) if checkpoint_dir is not None else None,
            "split": split,
            "sample_id": sample_id,
            "control_split": control_split,
            "control_sample_id": control_sample_id,
            "layer_selector": layer_selector,
            "mask_kind": mask_kind,
            "k": int(k),
            "alpha": float(alpha),
            "timestep": int(timestep),
            "noise_seed": int(noise_seed),
            "mask_seed": int(mask_seed),
            "prediction_type": prediction_type,
            "device": "cpu",
            "resolution": int(resolution),
        },
        "mask": {
            "mask_kind": mask_kind,
            "channel_indices": channel_indices,
            "k": int(len(channel_indices)),
            "alpha": float(alpha),
        },
        "metrics": {
            "selected_channel_abs_delta_pre": round(pre_selected_delta, 6),
            "selected_channel_abs_delta_post": round(post_selected_delta, 6),
            "selected_delta_retention_ratio": round(selected_delta_retention_ratio, 6),
            "off_mask_drift": round(off_mask_drift, 6),
        },
        "checks": {
            "records_written": len(records),
            "tensor_artifacts_written": int(sum(len(item) for item in artifact_paths.values())),
            "sample_pair_resolved": len(records) == 2,
            "device_cpu_only": True,
            "mask_applied": True,
        },
        "artifact_paths": {
            "summary": "summary.json",
            "records": "records.jsonl",
            "tensors_root": "tensors",
        },
        "notes": [
            "This scaffold applies the channel mask offline on captured activation tensors.",
            "It creates an executable packet surface for later bridge review, not a benchmark result.",
        ],
    }
    (workspace_path / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary


def export_gsa_observability_inmodel_packet(
    workspace: str | Path,
    repo_root: str | Path,
    assets_root: str | Path,
    checkpoint_root: str | Path,
    checkpoint_dir: str | Path | None,
    split: str,
    sample_id: str,
    control_split: str,
    control_sample_id: str,
    layer_selector: str,
    mask_kind: str = "top_abs_delta_k",
    k: int = 8,
    alpha: float = 0.5,
    timestep: int = 999,
    noise_seed: int = 0,
    mask_seed: int = 0,
    prediction_type: str = "epsilon",
    device: str = "cpu",
    resolution: int = 32,
    ddpm_num_steps: int = 1000,
    provenance_status: str = "workspace-verified",
    candidate: str = "I-C white-gray bridge packet",
) -> dict[str, Any]:
    if device.lower() != "cpu":
        raise ValueError("In-model packet export is CPU-first and does not authorize GPU use.")

    workspace_info = validate_gsa_workspace(repo_root)
    assets_path = Path(assets_root)
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    records_path = workspace_path / "records.jsonl"
    tensors_root = workspace_path / "tensors"
    tensors_root.mkdir(parents=True, exist_ok=True)

    selector = resolve_gsa_layer_selector(layer_selector, resolution=resolution)
    model, resolved_checkpoint_dir = _load_gsa_unet_checkpoint(
        checkpoint_root=checkpoint_root,
        checkpoint_dir=checkpoint_dir,
        resolution=resolution,
        device="cpu",
    )

    sample_specs = (
        {"split": split, "sample_id": sample_id, "role": "canary"},
        {"split": control_split, "sample_id": control_sample_id, "role": "control"},
    )
    baseline_by_role: dict[str, dict[str, torch.Tensor]] = {}
    bindings_by_role: dict[str, dict[str, str | None]] = {}
    for spec in sample_specs:
        binding = resolve_gsa_sample_binding(assets_path, split=spec["split"], sample_id=spec["sample_id"])
        bindings_by_role[spec["role"]] = binding
        sample_tensor = _load_image_tensor(binding["absolute_path"], resolution=resolution)
        noisy_sample = _prepare_noisy_sample(
            sample_tensor=sample_tensor,
            timestep=int(timestep),
            noise_seed=int(noise_seed),
            prediction_type=prediction_type,
            ddpm_num_steps=int(ddpm_num_steps),
        )
        baseline_forward = _run_layer_intervened_forward(
            model=model,
            layer_id=selector["layer_id"],
            noisy_sample=noisy_sample,
            timestep=int(timestep),
            channel_indices=None,
            alpha=1.0,
        )
        baseline_by_role[spec["role"]] = baseline_forward

    baseline_profiles = {
        role: _channelwise_profile(payload["post_activation"])
        for role, payload in baseline_by_role.items()
    }
    channel_indices = _select_mask_indices(
        canary_profile=baseline_profiles["canary"],
        control_profile=baseline_profiles["control"],
        mask_kind=mask_kind,
        k=k,
        random_seed=mask_seed,
    )
    selected_index_tensor = torch.tensor(channel_indices, dtype=torch.long)
    all_indices = torch.arange(int(baseline_profiles["canary"].shape[0]), dtype=torch.long)
    off_mask_indices = all_indices[~torch.isin(all_indices, selected_index_tensor)].tolist()

    intervened_by_role: dict[str, dict[str, torch.Tensor | bool]] = {}
    records: list[dict[str, Any]] = []
    prediction_rms_drift_by_role: dict[str, float] = {}
    prediction_max_abs_drift_by_role: dict[str, float] = {}
    for spec in sample_specs:
        role = str(spec["role"])
        binding = bindings_by_role[role]
        sample_tensor = _load_image_tensor(binding["absolute_path"], resolution=resolution)
        noisy_sample = _prepare_noisy_sample(
            sample_tensor=sample_tensor,
            timestep=int(timestep),
            noise_seed=int(noise_seed),
            prediction_type=prediction_type,
            ddpm_num_steps=int(ddpm_num_steps),
        )
        intervened_forward = _run_layer_intervened_forward(
            model=model,
            layer_id=selector["layer_id"],
            noisy_sample=noisy_sample,
            timestep=int(timestep),
            channel_indices=channel_indices,
            alpha=alpha,
        )
        intervened_by_role[role] = intervened_forward
        prediction_diff = (
            intervened_forward["prediction"].to(torch.float32)
            - baseline_by_role[role]["prediction"].to(torch.float32)
        )
        prediction_rms_drift = torch.sqrt(
            (
                prediction_diff.abs()
                ** 2
            ).mean()
        )
        prediction_rms_drift_by_role[role] = float(prediction_rms_drift.item())
        prediction_max_abs_drift_by_role[role] = float(prediction_diff.abs().max().item())

        sample_dir = tensors_root / _sanitize_path_fragment(str(binding["sample_id"]))
        sample_dir.mkdir(parents=True, exist_ok=True)
        baseline_layer_path = sample_dir / f"{_sanitize_path_fragment(layer_selector)}_t{int(timestep)}_baseline_layer.pt"
        intervened_layer_path = sample_dir / f"{_sanitize_path_fragment(layer_selector)}_t{int(timestep)}_intervened_layer.pt"
        baseline_prediction_path = sample_dir / f"{_sanitize_path_fragment(layer_selector)}_t{int(timestep)}_baseline_prediction.pt"
        intervened_prediction_path = sample_dir / f"{_sanitize_path_fragment(layer_selector)}_t{int(timestep)}_intervened_prediction.pt"
        torch.save(baseline_by_role[role]["post_activation"], baseline_layer_path)
        torch.save(intervened_forward["post_activation"], intervened_layer_path)
        torch.save(baseline_by_role[role]["prediction"], baseline_prediction_path)
        torch.save(intervened_forward["prediction"], intervened_prediction_path)
        records.append(
            {
                "split": spec["split"],
                "sample_id": binding["sample_id"],
                "role": role,
                "dataset_relpath": binding["dataset_relpath"],
                "layer_id": selector["layer_id"],
                "layer_selector": layer_selector,
                "timestep": int(timestep),
                "tensor_shape": list(baseline_by_role[role]["post_activation"].shape),
                "artifact_paths": {
                    "baseline_layer": baseline_layer_path.relative_to(workspace_path).as_posix(),
                    "intervened_layer": intervened_layer_path.relative_to(workspace_path).as_posix(),
                    "baseline_prediction": baseline_prediction_path.relative_to(workspace_path).as_posix(),
                    "intervened_prediction": intervened_prediction_path.relative_to(workspace_path).as_posix(),
                },
                "prediction_rms_drift": round(prediction_rms_drift_by_role[role], 12),
                "prediction_max_abs_drift": round(prediction_max_abs_drift_by_role[role], 12),
                "intervention_applied": bool(intervened_forward["intervention_applied"]),
            }
        )

    intervened_profiles = {
        role: _channelwise_profile(payload["post_activation"])
        for role, payload in intervened_by_role.items()
    }
    pre_selected_delta = float(
        torch.abs(
            baseline_profiles["canary"][selected_index_tensor] - baseline_profiles["control"][selected_index_tensor]
        ).mean().item()
    )
    post_selected_delta = float(
        torch.abs(
            intervened_profiles["canary"][selected_index_tensor] - intervened_profiles["control"][selected_index_tensor]
        ).mean().item()
    )
    selected_delta_retention_ratio = float(post_selected_delta / pre_selected_delta) if pre_selected_delta else 0.0

    off_mask_drift_values: list[float] = []
    if off_mask_indices:
        off_mask_index_tensor = torch.tensor(off_mask_indices, dtype=torch.long)
        for role in ("canary", "control"):
            drift = torch.abs(
                intervened_profiles[role][off_mask_index_tensor] - baseline_profiles[role][off_mask_index_tensor]
            ).mean()
            off_mask_drift_values.append(float(drift.item()))
    off_mask_drift = float(sum(off_mask_drift_values) / len(off_mask_drift_values)) if off_mask_drift_values else 0.0
    mean_prediction_rms_drift = float(
        sum(prediction_rms_drift_by_role.values()) / len(prediction_rms_drift_by_role)
    )
    mean_prediction_max_abs_drift = float(
        sum(prediction_max_abs_drift_by_role.values()) / len(prediction_max_abs_drift_by_role)
    )

    records_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema": "diffaudit.gsa_observability.inmodel_packet.v1",
        "status": "ready",
        "track": "white-box",
        "method": "gsa-observability",
        "mode": "inmodel-packet-export",
        "candidate": candidate,
        "contract_stage": "stage-3-inmodel-intervention-surface",
        "provenance_status": provenance_status,
        "gpu_release": "none",
        "admitted_change": "none",
        "workspace": workspace_info,
        "requested": {
            "workspace": str(workspace_path),
            "assets_root": str(assets_path),
            "checkpoint_root": str(Path(checkpoint_root)),
            "checkpoint_dir": str(Path(checkpoint_dir)) if checkpoint_dir is not None else None,
            "split": split,
            "sample_id": sample_id,
            "control_split": control_split,
            "control_sample_id": control_sample_id,
            "layer_selector": layer_selector,
            "mask_kind": mask_kind,
            "k": int(k),
            "alpha": float(alpha),
            "timestep": int(timestep),
            "noise_seed": int(noise_seed),
            "mask_seed": int(mask_seed),
            "prediction_type": prediction_type,
            "device": "cpu",
            "resolution": int(resolution),
        },
        "mask": {
            "mask_kind": mask_kind,
            "channel_indices": channel_indices,
            "k": int(len(channel_indices)),
            "alpha": float(alpha),
        },
        "metrics": {
            "selected_channel_abs_delta_pre": round(pre_selected_delta, 6),
            "selected_channel_abs_delta_post": round(post_selected_delta, 6),
            "selected_delta_retention_ratio": round(selected_delta_retention_ratio, 6),
            "off_mask_drift": round(off_mask_drift, 6),
            "epsilon_prediction_rms_drift_mean": round(mean_prediction_rms_drift, 12),
            "epsilon_prediction_max_abs_drift_mean": round(mean_prediction_max_abs_drift, 12),
            "canary_prediction_rms_drift": round(prediction_rms_drift_by_role["canary"], 12),
            "control_prediction_rms_drift": round(prediction_rms_drift_by_role["control"], 12),
            "canary_prediction_max_abs_drift": round(prediction_max_abs_drift_by_role["canary"], 12),
            "control_prediction_max_abs_drift": round(prediction_max_abs_drift_by_role["control"], 12),
            "prediction_drift_gap": round(
                prediction_rms_drift_by_role["canary"] - prediction_rms_drift_by_role["control"],
                12,
            ),
        },
        "checks": {
            "records_written": len(records),
            "tensor_artifacts_written": int(
                sum(len(item["artifact_paths"]) for item in records)
            ),
            "sample_pair_resolved": len(records) == 2,
            "device_cpu_only": True,
            "intervention_applied": all(bool(item["intervention_applied"]) for item in records),
            "baseline_and_intervened_forward_complete": True,
        },
        "artifact_paths": {
            "summary": "summary.json",
            "records": "records.jsonl",
            "tensors_root": "tensors",
        },
        "notes": [
            "This packet applies the selected channel mask inside the hooked forward path.",
            "The exported prediction drift is a white-box downstream canary, not cross-box support by itself.",
        ],
    }
    (workspace_path / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary
