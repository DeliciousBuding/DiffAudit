"""Runtime helpers for the white-box GSA DDPM workflow."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REQUIRED_GSA_WORKSPACE_FILES = (
    "DDPM/gen_l2_gradients_DDPM.py",
    "DDPM/train_unconditional.py",
    "test_attack_accuracy.py",
)


def validate_gsa_workspace(repo_root: str | Path) -> dict[str, str]:
    repo_path = Path(repo_root)
    missing = [
        relative_path
        for relative_path in REQUIRED_GSA_WORKSPACE_FILES
        if not (repo_path / relative_path).exists()
    ]
    if missing:
        raise FileNotFoundError(f"GSA workspace is missing required files: {', '.join(missing)}")
    return {
        "status": "ready",
        "repo_root": str(repo_path),
        "gradient_entrypoint": str(repo_path / "DDPM" / "gen_l2_gradients_DDPM.py"),
        "attack_entrypoint": str(repo_path / "test_attack_accuracy.py"),
    }


def _latest_checkpoint_dir(checkpoint_root: str | Path) -> Path:
    root = Path(checkpoint_root)
    candidates = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")],
        key=lambda path: int(path.name.split("-")[1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directory found in {root}")
    return candidates[-1]


def _safe_latest_checkpoint_dir(checkpoint_root: str | Path) -> Path | None:
    try:
        return _latest_checkpoint_dir(checkpoint_root)
    except FileNotFoundError:
        return None


def _dataset_has_files(dataset_dir: str | Path) -> bool:
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        return False
    return bool(_iter_dataset_files(dataset_path))


def _discover_shadow_specs(
    dataset_root: str | Path,
    checkpoint_root: str | Path,
) -> list[dict[str, str]]:
    dataset_path = Path(dataset_root)
    checkpoint_path = Path(checkpoint_root)

    legacy_member = dataset_path / "shadow-member"
    legacy_nonmember = dataset_path / "shadow-nonmember"
    legacy_checkpoint = checkpoint_path / "shadow"
    if legacy_member.exists() and legacy_nonmember.exists() and legacy_checkpoint.exists():
        return [
            {
                "name": "shadow",
                "member_dataset": str(legacy_member),
                "nonmember_dataset": str(legacy_nonmember),
                "checkpoint_dir": str(legacy_checkpoint),
            }
        ]

    discovered: list[dict[str, str]] = []
    for member_dir in sorted(dataset_path.glob("shadow-*-member")):
        prefix = member_dir.name[: -len("-member")]
        discovered.append(
            {
                "name": prefix,
                "member_dataset": str(member_dir),
                "nonmember_dataset": str(dataset_path / f"{prefix}-nonmember"),
                "checkpoint_dir": str(checkpoint_path / prefix),
            }
        )
    return discovered


def probe_gsa_assets(
    assets_root: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    workspace = validate_gsa_workspace(repo_root)
    assets_path = Path(assets_root)
    dataset_root = assets_path / "datasets"
    checkpoint_root = assets_path / "checkpoints"
    manifest_root = assets_path / "manifests"
    source_root = assets_path / "sources"
    shadow_specs = _discover_shadow_specs(dataset_root, checkpoint_root)

    target_checkpoint = checkpoint_root / "target"
    target_latest = _safe_latest_checkpoint_dir(target_checkpoint) if target_checkpoint.exists() else None
    manifest_files = [path for path in manifest_root.iterdir()] if manifest_root.exists() else []
    shadow_latest_dirs: list[str] = []
    for spec in shadow_specs:
        checkpoint_dir = Path(spec["checkpoint_dir"])
        latest = _safe_latest_checkpoint_dir(checkpoint_dir) if checkpoint_dir.exists() else None
        if latest is not None:
            shadow_latest_dirs.append(str(latest))

    checks = {
        "workspace_files": True,
        "assets_root": assets_path.exists(),
        "target_member_dataset": _dataset_has_files(dataset_root / "target-member"),
        "target_nonmember_dataset": _dataset_has_files(dataset_root / "target-nonmember"),
        "target_checkpoint": bool(target_latest and target_latest.exists()),
        "shadow_assets": bool(shadow_specs),
        "shadow_datasets": bool(shadow_specs)
        and all(
            _dataset_has_files(spec["member_dataset"]) and _dataset_has_files(spec["nonmember_dataset"])
            for spec in shadow_specs
        ),
        "shadow_checkpoints": bool(shadow_specs)
        and len(shadow_latest_dirs) == len(shadow_specs),
        "manifest_dir": manifest_root.exists(),
        "manifest_file": bool(manifest_files),
        "sources_dir": source_root.exists(),
    }
    labels = {
        "workspace_files": "gsa workspace files",
        "assets_root": "assets root",
        "target_member_dataset": "target-member dataset",
        "target_nonmember_dataset": "target-nonmember dataset",
        "target_checkpoint": "target checkpoint-*",
        "shadow_assets": "shadow asset specs",
        "shadow_datasets": "shadow datasets",
        "shadow_checkpoints": "shadow checkpoint-*",
        "manifest_dir": "manifests dir",
        "manifest_file": "manifest file",
        "sources_dir": "sources dir",
    }
    paths = {
        "repo_root": str(Path(repo_root)),
        "assets_root": str(assets_path),
        "target_member_dataset": str(dataset_root / "target-member"),
        "target_nonmember_dataset": str(dataset_root / "target-nonmember"),
        "target_checkpoint": str(target_latest) if target_latest else str(target_checkpoint),
        "shadow_specs": shadow_specs,
        "shadow_checkpoints": shadow_latest_dirs,
        "manifests": str(manifest_root),
        "sources": str(source_root),
    }
    missing_keys = [name for name, ready in checks.items() if not ready]
    missing: list[str] = []
    for name in missing_keys:
        value = paths.get(name, labels[name])
        if isinstance(value, list):
            if value:
                missing.extend(str(item) for item in value)
            else:
                missing.append(labels[name])
        else:
            missing.append(str(value))
    return {
        "status": "ready" if all(checks.values()) else "blocked",
        "checks": checks,
        "paths": paths,
        "missing_keys": missing_keys,
        "missing": missing,
        "missing_items": [Path(item).name if Path(item).name else item for item in missing],
        "missing_description": " / ".join(labels[name] for name in missing_keys),
        "workspace": workspace,
    }


def _run_subprocess(command: list[str], cwd: str | Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=str(Path(cwd)),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    return {
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout.strip().splitlines()[-5:] if completed.stdout.strip() else [],
        "stderr_tail": completed.stderr.strip().splitlines()[-5:] if completed.stderr.strip() else [],
        "command": command,
    }


def _load_gradient_tensor(path: str | Path) -> torch.Tensor:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"GSA gradient payload must be a torch.Tensor: {path}")
    return payload.cpu()


def _extract_into_tensor(arr: torch.Tensor | np.ndarray, timesteps: torch.Tensor, broadcast_shape: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def _resolve_runtime_device(device: str) -> str:
    requested_device = device.lower()
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported GSA runtime device: {device}")
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested_device


def _iter_dataset_files(dataset_dir: str | Path) -> list[Path]:
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted(
        path
        for path in Path(dataset_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in allowed_suffixes
    )


def _filter_dataset_files_by_sample_ids(
    dataset_files: list[Path],
    sample_id_allowlist: list[int] | None,
) -> list[Path]:
    if not sample_id_allowlist:
        return list(dataset_files)
    allowed = {int(value) for value in sample_id_allowlist}
    selected: list[Path] = []
    for path in dataset_files:
        match = re.search(r"-(\d+)\.[^.]+$", path.name)
        if match is None:
            continue
        if int(match.group(1)) in allowed:
            selected.append(path)
    return selected


def _load_gsa_image_tensor(
    image_path: str | Path,
    resolution: int,
    device: str,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB").resize((int(resolution), int(resolution)))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0).to(device)


def _draw_loss_score_noise(
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    noise_seed: int | None,
    sample_key: str,
) -> torch.Tensor:
    if noise_seed is None:
        return torch.randn(shape, device=device, dtype=dtype)

    seed_material = f"{int(noise_seed)}:{sample_key}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big", signed=False)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(derived_seed)
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def _load_frozen_mask_summary(mask_summary: str | Path) -> dict[str, Any]:
    summary_path = Path(mask_summary)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("mode") != "inmodel-packet-export":
        raise ValueError(f"Frozen mask summary must come from inmodel-packet-export: {summary_path}")
    mask = payload.get("mask") or {}
    requested = payload.get("requested") or {}
    channel_indices = [int(index) for index in mask.get("channel_indices", [])]
    if not channel_indices:
        raise ValueError(f"Frozen mask summary is missing channel indices: {summary_path}")
    return {
        "summary_path": str(summary_path.resolve()),
        "requested": requested,
        "mask": {
            "mask_kind": str(mask.get("mask_kind", "top_abs_delta_k")),
            "channel_indices": channel_indices,
            "k": int(mask.get("k", len(channel_indices))),
            "alpha": float(mask.get("alpha", 1.0)),
        },
        "locality_anchor_metrics": payload.get("metrics", {}),
    }


def _extract_gsa_gradients_with_fixed_mask(
    *,
    dataset_dir: str | Path,
    checkpoint_root: str | Path,
    checkpoint_dir: str | Path | None,
    output_path: str | Path,
    layer_selector: str,
    channel_indices: list[int] | None,
    alpha: float,
    resolution: int,
    ddpm_num_steps: int,
    sampling_frequency: int,
    attack_method: int,
    prediction_type: str,
    device: str,
    extraction_max_samples: int | None = None,
) -> dict[str, Any]:
    from diffaudit.attacks.gsa_observability import (
        _apply_channel_mask,
        _build_gsa_noise_scheduler,
        _extract_model_prediction,
        _load_gsa_unet_checkpoint,
        resolve_gsa_layer_selector,
    )

    resolved_device = _resolve_runtime_device(device)
    model, resolved_checkpoint_dir = _load_gsa_unet_checkpoint(
        checkpoint_root=checkpoint_root,
        checkpoint_dir=checkpoint_dir,
        resolution=resolution,
        device=resolved_device,
    )
    selector = resolve_gsa_layer_selector(layer_selector, resolution=resolution)
    modules = dict(model.named_modules())
    if selector["layer_id"] not in modules:
        raise KeyError(f"Layer id not found in model: {selector['layer_id']}")
    scheduler = _build_gsa_noise_scheduler(
        ddpm_num_steps=ddpm_num_steps,
        prediction_type=prediction_type,
    )
    dataset_files = _iter_dataset_files(dataset_dir)
    if extraction_max_samples is not None:
        dataset_files = dataset_files[: int(extraction_max_samples)]
    timestep_stride = max(1, int(ddpm_num_steps / sampling_frequency))
    timestep_values = [x - 1 for x in range(timestep_stride, ddpm_num_steps + 1, timestep_stride)]
    all_sample_grads: list[torch.Tensor] = []

    for image_path in dataset_files:
        clean_image = _load_gsa_image_tensor(image_path=image_path, resolution=resolution, device=resolved_device)
        clean_batch = clean_image.repeat(len(timestep_values), 1, 1, 1)
        noise = torch.randn(clean_batch.shape, device=clean_batch.device, dtype=clean_batch.dtype)
        timesteps = torch.tensor(timestep_values, device=clean_batch.device).long()
        noisy_images = scheduler.add_noise(clean_batch, noise, timesteps)
        model.zero_grad(set_to_none=True)

        def _hook(_: torch.nn.Module, __: tuple[torch.Tensor, ...], output: Any):
            if not channel_indices:
                return output
            tensor = output[0] if isinstance(output, tuple) else output
            masked = _apply_channel_mask(tensor, channel_indices=channel_indices, alpha=alpha)
            if isinstance(output, tuple):
                return (masked, *output[1:])
            return masked

        handle = modules[selector["layer_id"]].register_forward_hook(_hook)
        try:
            model_output = model(noisy_images, timesteps)
        finally:
            handle.remove()

        prediction = _extract_model_prediction(model_output)
        if attack_method == 1:
            if prediction_type == "epsilon":
                loss = F.mse_loss(prediction, noise)
            elif prediction_type == "sample":
                alpha_t = _extract_into_tensor(
                    scheduler.alphas_cumprod,
                    timesteps,
                    tuple(clean_batch.shape),
                )
                snr_weights = alpha_t / (1 - alpha_t)
                loss = snr_weights * F.mse_loss(prediction, clean_batch, reduction="none")
                loss = loss.mean()
            else:
                raise ValueError(f"Unsupported prediction type: {prediction_type}")
            loss.backward()
            gradients = torch.cat(
                [
                    torch.norm(parameter.grad).unsqueeze(0)
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
            )
            all_sample_grads.append(gradients.detach().cpu().unsqueeze(0))
            model.zero_grad(set_to_none=True)
        elif attack_method == 2:
            per_timestep_grads: list[torch.Tensor] = []
            for index in range(len(timesteps)):
                if prediction_type == "epsilon":
                    loss = F.mse_loss(prediction[index].unsqueeze(0), noise[index].unsqueeze(0))
                elif prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        scheduler.alphas_cumprod,
                        timesteps,
                        tuple(clean_batch.shape),
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = snr_weights * F.mse_loss(prediction, clean_batch, reduction="none")
                    loss = loss.mean()
                else:
                    raise ValueError(f"Unsupported prediction type: {prediction_type}")
                loss.backward(retain_graph=True)
                gradients = torch.cat(
                    [
                        torch.norm(parameter.grad).unsqueeze(0)
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    ]
                )
                per_timestep_grads.append(gradients.detach().cpu())
                model.zero_grad(set_to_none=True)
            all_sample_grads.append(torch.stack(per_timestep_grads).mean(dim=0).unsqueeze(0))
        else:
            raise ValueError(f"Unsupported GSA attack method for intervention review: {attack_method}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gradient_tensor = torch.cat(all_sample_grads) if all_sample_grads else torch.empty((0, 0), dtype=torch.float32)
    torch.save(gradient_tensor, output_path)
    return {
        "status": "ready",
        "output_path": str(output_path),
        "checkpoint_dir": str(resolved_checkpoint_dir),
        "sample_count": len(dataset_files),
        "intervention_enabled": bool(channel_indices),
    }


def _extract_gsa_loss_scores(
    *,
    dataset_dir: str | Path,
    checkpoint_root: str | Path,
    checkpoint_dir: str | Path | None,
    output_path: str | Path,
    records_path: str | Path,
    resolution: int,
    ddpm_num_steps: int,
    sampling_frequency: int,
    attack_method: int,
    prediction_type: str,
    device: str,
    extraction_max_samples: int | None = None,
    sample_id_allowlist: list[int] | None = None,
    noise_seed: int | None = None,
) -> dict[str, Any]:
    from diffaudit.attacks.gsa_observability import (
        _build_gsa_noise_scheduler,
        _extract_model_prediction,
        _load_gsa_unet_checkpoint,
    )

    resolved_device = _resolve_runtime_device(device)
    model, resolved_checkpoint_dir = _load_gsa_unet_checkpoint(
        checkpoint_root=checkpoint_root,
        checkpoint_dir=checkpoint_dir,
        resolution=resolution,
        device=resolved_device,
    )
    scheduler = _build_gsa_noise_scheduler(
        ddpm_num_steps=ddpm_num_steps,
        prediction_type=prediction_type,
    )
    dataset_root = Path(dataset_dir)
    dataset_files = _iter_dataset_files(dataset_root)
    dataset_files = _filter_dataset_files_by_sample_ids(
        dataset_files,
        sample_id_allowlist=sample_id_allowlist,
    )
    if extraction_max_samples is not None:
        dataset_files = dataset_files[: int(extraction_max_samples)]
    timestep_stride = max(1, int(ddpm_num_steps / sampling_frequency))
    timestep_values = [x - 1 for x in range(timestep_stride, ddpm_num_steps + 1, timestep_stride)]

    score_values: list[float] = []
    records: list[dict[str, Any]] = []
    model.eval()
    for index, image_path in enumerate(dataset_files):
        clean_image = _load_gsa_image_tensor(image_path=image_path, resolution=resolution, device=resolved_device)
        clean_batch = clean_image.repeat(len(timestep_values), 1, 1, 1)
        sample_key = image_path.relative_to(dataset_root).as_posix()
        noise = _draw_loss_score_noise(
            shape=tuple(clean_batch.shape),
            device=clean_batch.device,
            dtype=clean_batch.dtype,
            noise_seed=noise_seed,
            sample_key=sample_key,
        )
        timesteps = torch.tensor(timestep_values, device=clean_batch.device).long()
        noisy_images = scheduler.add_noise(clean_batch, noise, timesteps)

        with torch.no_grad():
            model_output = model(noisy_images, timesteps)
            prediction = _extract_model_prediction(model_output)
            if attack_method == 1:
                if prediction_type == "epsilon":
                    loss = F.mse_loss(prediction, noise)
                elif prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        scheduler.alphas_cumprod,
                        timesteps,
                        tuple(clean_batch.shape),
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = snr_weights * F.mse_loss(prediction, clean_batch, reduction="none")
                    loss = loss.mean()
                else:
                    raise ValueError(f"Unsupported prediction type: {prediction_type}")
                score = float(loss.detach().cpu())
            elif attack_method == 2:
                per_timestep_losses: list[torch.Tensor] = []
                for timestep_index in range(len(timesteps)):
                    if prediction_type == "epsilon":
                        loss = F.mse_loss(
                            prediction[timestep_index].unsqueeze(0),
                            noise[timestep_index].unsqueeze(0),
                        )
                    elif prediction_type == "sample":
                        alpha_t = _extract_into_tensor(
                            scheduler.alphas_cumprod,
                            timesteps[timestep_index].unsqueeze(0),
                            tuple(clean_image.shape),
                        )
                        snr_weights = alpha_t / (1 - alpha_t)
                        loss = snr_weights * F.mse_loss(
                            prediction[timestep_index].unsqueeze(0),
                            clean_image,
                            reduction="none",
                        )
                        loss = loss.mean()
                    else:
                        raise ValueError(f"Unsupported prediction type: {prediction_type}")
                    per_timestep_losses.append(loss.detach().cpu())
                score = float(torch.stack(per_timestep_losses).mean())
            else:
                raise ValueError(f"Unsupported GSA attack method for loss-score export: {attack_method}")

        score_values.append(score)
        records.append(
            {
                "sample_index": index,
                "sample_path": sample_key,
                "loss_score": score,
            }
        )

    output_path = Path(output_path)
    records_path = Path(records_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    score_tensor = torch.tensor(score_values, dtype=torch.float32)
    torch.save(score_tensor, output_path)
    records_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records),
        encoding="utf-8",
    )
    score_stats = {
        "mean": round(float(score_tensor.mean().item()), 6) if score_values else None,
        "min": round(float(score_tensor.min().item()), 6) if score_values else None,
        "max": round(float(score_tensor.max().item()), 6) if score_values else None,
    }
    return {
        "status": "ready",
        "output_path": str(output_path),
        "records_path": str(records_path),
        "checkpoint_dir": str(resolved_checkpoint_dir),
        "sample_count": len(dataset_files),
        "score_stats": score_stats,
        "noise_seed": int(noise_seed) if noise_seed is not None else None,
    }


def _max_tpr_under_fpr_cap(
    fpr: np.ndarray,
    tpr: np.ndarray,
    *,
    cap: float,
) -> float:
    valid = tpr[fpr <= cap]
    if valid.size == 0:
        return 0.0
    return float(valid.max())


def _score_metric_bundle(
    member_scores: list[float],
    nonmember_scores: list[float],
    *,
    score_direction: str | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score, roc_curve

    member_array = np.asarray(member_scores, dtype=np.float64)
    nonmember_array = np.asarray(nonmember_scores, dtype=np.float64)
    if member_array.size == 0 or nonmember_array.size == 0:
        raise ValueError("Both member and nonmember score lists must be non-empty")

    inferred_direction = "member-higher" if float(member_array.mean()) >= float(nonmember_array.mean()) else "member-lower"
    applied_direction = score_direction or inferred_direction
    if applied_direction not in {"member-higher", "member-lower"}:
        raise ValueError(f"Unsupported score direction: {applied_direction}")

    raw_scores = np.concatenate([member_array, nonmember_array])
    labels = np.concatenate(
        [
            np.ones(member_array.size, dtype=np.int64),
            np.zeros(nonmember_array.size, dtype=np.int64),
        ]
    )
    oriented_scores = raw_scores if applied_direction == "member-higher" else -raw_scores

    auc = float(roc_auc_score(labels, oriented_scores))
    fpr, tpr, thresholds = roc_curve(labels, oriented_scores)
    if threshold is None:
        finite_indices = np.flatnonzero(np.isfinite(thresholds))
        candidate_indices = finite_indices if finite_indices.size else np.arange(thresholds.shape[0])
        best_index = int(candidate_indices[np.argmax((tpr - fpr)[candidate_indices])])
        applied_threshold = float(thresholds[best_index])
    else:
        applied_threshold = float(threshold)

    predictions = (oriented_scores >= applied_threshold).astype(np.int64)
    asr = float(np.mean(predictions == labels))
    return {
        "auc": round(auc, 6),
        "asr": round(asr, 6),
        "tpr_at_1pct_fpr": round(_max_tpr_under_fpr_cap(fpr, tpr, cap=0.01), 6),
        "tpr_at_0_1pct_fpr": round(_max_tpr_under_fpr_cap(fpr, tpr, cap=0.001), 6),
        "member_mean_score": round(float(member_array.mean()), 6),
        "nonmember_mean_score": round(float(nonmember_array.mean()), 6),
        "member_eval_size": int(member_array.size),
        "nonmember_eval_size": int(nonmember_array.size),
        "score_direction": applied_direction,
        "inferred_score_direction": inferred_direction,
        "applied_threshold": applied_threshold,
        "threshold_selection_rule": "shadow-youden-j" if threshold is None else "frozen-transfer-threshold",
    }


def _fit_univariate_gaussian(scores: list[float], *, variance_floor: float = 1e-6) -> dict[str, float]:
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.size == 0:
        raise ValueError("Gaussian density fit requires at least one score")
    if score_array.size == 1:
        variance = variance_floor
    else:
        variance = max(float(score_array.var(ddof=1)), variance_floor)
    return {
        "mean": float(score_array.mean()),
        "variance": float(variance),
    }


def _gaussian_log_likelihood_ratio(
    scores: list[float],
    *,
    member_fit: dict[str, float],
    nonmember_fit: dict[str, float],
) -> list[float]:
    score_array = np.asarray(scores, dtype=np.float64)
    member_mean = float(member_fit["mean"])
    member_var = float(member_fit["variance"])
    nonmember_mean = float(nonmember_fit["mean"])
    nonmember_var = float(nonmember_fit["variance"])
    member_logpdf = -0.5 * np.log(2.0 * np.pi * member_var) - ((score_array - member_mean) ** 2) / (2.0 * member_var)
    nonmember_logpdf = -0.5 * np.log(2.0 * np.pi * nonmember_var) - ((score_array - nonmember_mean) ** 2) / (
        2.0 * nonmember_var
    )
    return [float(value) for value in (member_logpdf - nonmember_logpdf).tolist()]


def _load_gsa_loss_score_export_summary(summary_path: str | Path) -> dict[str, Any]:
    resolved_summary = Path(summary_path)
    payload = json.loads(resolved_summary.read_text(encoding="utf-8"))
    if payload.get("mode") != "loss-score-export":
        raise ValueError(f"GSA loss-score threshold evaluator requires a loss-score-export summary: {resolved_summary}")
    if payload.get("status") != "ready":
        raise ValueError(f"GSA loss-score threshold evaluator requires a ready export summary: {resolved_summary}")
    return payload


def _flatten_score_tensor(path: str | Path) -> list[float]:
    tensor = _load_gradient_tensor(path)
    return [float(value) for value in tensor.reshape(-1).tolist()]


def evaluate_gsa_loss_score_packet(
    workspace: str | Path,
    packet_summary: str | Path,
    evaluation_style: str = "threshold-transfer",
    provenance_status: str = "workspace-verified",
) -> dict[str, Any]:
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    export_payload = _load_gsa_loss_score_export_summary(packet_summary)
    exports = export_payload["exports"]
    shadow_specs = export_payload.get("artifact_paths", {}).get("shadow_specs", [])

    shadow_member_groups: list[list[float]] = []
    shadow_nonmember_groups: list[list[float]] = []
    shadow_records: list[dict[str, Any]] = []
    shadow_score_files_ready = True

    for spec in shadow_specs:
        job_prefix = spec["name"].replace("-", "_")
        member_export = exports[f"{job_prefix}_member"]
        nonmember_export = exports[f"{job_prefix}_non_member"]
        member_path = member_export["output_path"]
        nonmember_path = nonmember_export["output_path"]
        shadow_score_files_ready = shadow_score_files_ready and Path(member_path).exists() and Path(nonmember_path).exists()
        member_scores = _flatten_score_tensor(member_path)
        nonmember_scores = _flatten_score_tensor(nonmember_path)
        shadow_member_groups.append(member_scores)
        shadow_nonmember_groups.append(nonmember_scores)
        shadow_records.append(
            {
                "shadow_name": spec["name"],
                "member_scores_path": member_path,
                "nonmember_scores_path": nonmember_path,
                "member_eval_size": len(member_scores),
                "nonmember_eval_size": len(nonmember_scores),
            }
        )

    target_member_path = exports["target_member"]["output_path"]
    target_nonmember_path = exports["target_non_member"]["output_path"]
    target_member_scores = _flatten_score_tensor(target_member_path)
    target_nonmember_scores = _flatten_score_tensor(target_nonmember_path)

    pooled_shadow_member_scores = [score for group in shadow_member_groups for score in group]
    pooled_shadow_nonmember_scores = [score for group in shadow_nonmember_groups for score in group]

    shadow_fit: dict[str, Any] | None = None
    if evaluation_style == "threshold-transfer":
        shadow_metrics = _score_metric_bundle(
            pooled_shadow_member_scores,
            pooled_shadow_nonmember_scores,
        )
        target_transfer_metrics = _score_metric_bundle(
            target_member_scores,
            target_nonmember_scores,
            score_direction=shadow_metrics["score_direction"],
            threshold=shadow_metrics["applied_threshold"],
        )
        target_self_metrics = _score_metric_bundle(
            target_member_scores,
            target_nonmember_scores,
        )
        mode = "loss-score-threshold-eval"
        orientation_source = "pooled-shadow-only"
        threshold_source = "pooled-shadow-only"
        evaluation_notes = [
            "Orientation and threshold are frozen from pooled shadow member/nonmember loss scores only, then transferred to target evaluation.",
            "The target self-board is emitted only as a diagnostic reference and must not be promoted as the packet verdict.",
            "Bounded extraction_max_samples packets remain below release-grade low-FPR honesty until larger honest packets are executed.",
        ]
    elif evaluation_style == "gaussian-likelihood-ratio-transfer":
        shadow_member_fit = _fit_univariate_gaussian(pooled_shadow_member_scores)
        shadow_nonmember_fit = _fit_univariate_gaussian(pooled_shadow_nonmember_scores)
        shadow_fit = {
            "member": {
                "distribution": "gaussian",
                "mean": round(shadow_member_fit["mean"], 6),
                "variance": round(shadow_member_fit["variance"], 6),
            },
            "nonmember": {
                "distribution": "gaussian",
                "mean": round(shadow_nonmember_fit["mean"], 6),
                "variance": round(shadow_nonmember_fit["variance"], 6),
            },
        }
        shadow_member_llr = _gaussian_log_likelihood_ratio(
            pooled_shadow_member_scores,
            member_fit=shadow_member_fit,
            nonmember_fit=shadow_nonmember_fit,
        )
        shadow_nonmember_llr = _gaussian_log_likelihood_ratio(
            pooled_shadow_nonmember_scores,
            member_fit=shadow_member_fit,
            nonmember_fit=shadow_nonmember_fit,
        )
        target_member_llr = _gaussian_log_likelihood_ratio(
            target_member_scores,
            member_fit=shadow_member_fit,
            nonmember_fit=shadow_nonmember_fit,
        )
        target_nonmember_llr = _gaussian_log_likelihood_ratio(
            target_nonmember_scores,
            member_fit=shadow_member_fit,
            nonmember_fit=shadow_nonmember_fit,
        )
        shadow_metrics = _score_metric_bundle(
            shadow_member_llr,
            shadow_nonmember_llr,
        )
        target_transfer_metrics = _score_metric_bundle(
            target_member_llr,
            target_nonmember_llr,
            score_direction=shadow_metrics["score_direction"],
            threshold=shadow_metrics["applied_threshold"],
        )
        target_self_metrics = _score_metric_bundle(
            target_member_llr,
            target_nonmember_llr,
        )
        mode = "loss-score-lr-eval"
        orientation_source = "pooled-shadow-gaussian-lr"
        threshold_source = "pooled-shadow-gaussian-lr"
        evaluation_notes = [
            "Likelihood-ratio scores are derived from shadow-only Gaussian density fits on member/nonmember loss-score pools.",
            "The target transfer board reuses the frozen shadow-side likelihood-ratio threshold; the target self-board remains diagnostic-only.",
            "This bounded scorer extension reuses the same packet identity and runtime schedule as the frozen threshold-transfer packet.",
        ]
    else:
        raise ValueError(f"Unsupported evaluation_style: {evaluation_style}")

    checks = {
        "packet_summary_ready": True,
        "target_score_files": Path(target_member_path).exists() and Path(target_nonmember_path).exists(),
        "shadow_score_files": shadow_score_files_ready,
        "shadow_groups_present": bool(shadow_specs),
    }
    result = {
        "status": "ready" if all(checks.values()) else "error",
        "track": "white-box",
        "method": "gsa",
        "paper": "WhiteBox_GSA_PoPETS2025",
        "mode": mode,
        "workspace": str(workspace_path),
        "workspace_name": workspace_path.name,
        "contract_stage": "target",
        "asset_grade": export_payload.get("asset_grade", "real-asset-closed-loop"),
        "provenance_status": provenance_status,
        "evidence_level": (
            "bounded-loss-score-threshold-eval"
            if evaluation_style == "threshold-transfer"
            else "bounded-loss-score-lr-eval"
        ),
        "checks": checks,
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
            "packet_summary": str(Path(packet_summary).resolve()),
            "target_member_scores": target_member_path,
            "target_nonmember_scores": target_nonmember_path,
        },
        "runtime": {
            "packet_runtime": export_payload.get("runtime", {}),
            "evaluation_style": evaluation_style,
            "orientation_source": orientation_source,
            "threshold_source": threshold_source,
            "target_self_board": "diagnostic-only",
        },
        "shadow_pool": {
            "group_count": len(shadow_specs),
            "shadow_specs": shadow_records,
            "metrics": shadow_metrics,
        },
        "target_transfer": {
            "metrics": target_transfer_metrics,
        },
        "target_self_diagnostic": {
            "metrics": target_self_metrics,
        },
        "notes": evaluation_notes,
    }
    if shadow_fit is not None:
        result["shadow_pool"]["density_fit"] = shadow_fit
    (workspace_path / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return result


def _truncate_samples(tensor: torch.Tensor, max_samples: int | None) -> torch.Tensor:
    if max_samples is None:
        return tensor
    return tensor[:max_samples]


def _preprocess(
    member: torch.Tensor,
    nonmember: torch.Tensor,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    member = _truncate_samples(member, max_samples)
    nonmember = _truncate_samples(nonmember, max_samples)
    train_np = member.cpu().numpy()
    test_np = nonmember.cpu().numpy()
    train_np = train_np[0 : test_np.shape[0]]
    train_y_np = np.ones(train_np.shape[0], dtype=np.int64)
    test_y_np = np.zeros(test_np.shape[0], dtype=np.int64)
    x = np.vstack((train_np, test_np))
    y = np.concatenate((train_y_np, test_y_np))
    from sklearn import preprocessing

    x = preprocessing.scale(x)
    return x, y


def _evaluate_gsa_closed_loop(
    target_member_path: str | Path,
    target_nonmember_path: str | Path,
    shadow_member_paths: list[str | Path],
    shadow_nonmember_paths: list[str | Path],
    max_samples: int | None = None,
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    shadow_member = torch.cat([_load_gradient_tensor(path) for path in shadow_member_paths], dim=0)
    shadow_nonmember = torch.cat([_load_gradient_tensor(path) for path in shadow_nonmember_paths], dim=0)
    target_member = _load_gradient_tensor(target_member_path)
    target_nonmember = _load_gradient_tensor(target_nonmember_path)

    shadow_x, shadow_y = _preprocess(shadow_member, shadow_nonmember, max_samples=max_samples)
    shadow_train_x, _, shadow_train_y, _ = train_test_split(
        shadow_x,
        shadow_y,
        test_size=0.3,
        random_state=0,
        stratify=shadow_y,
    )
    xgb = XGBClassifier(
        n_estimators=200,
        random_state=0,
        eval_metric="logloss",
    )
    xgb.fit(shadow_train_x, shadow_train_y)

    target_x, target_y = _preprocess(target_member, target_nonmember, max_samples=max_samples)
    pred = xgb.predict(target_x)
    pred_prob = xgb.predict_proba(target_x)[:, 1]
    roc_auc = float(roc_auc_score(target_y, pred_prob))
    fpr, tpr, _ = roc_curve(target_y, pred_prob)
    return {
        "auc": round(roc_auc, 6),
        "asr": round(float(accuracy_score(target_y, pred)), 6),
        "tpr_at_1pct_fpr": round(float(tpr[np.argmin(np.abs(fpr - 0.01))]), 6),
        "tpr_at_0_1pct_fpr": round(float(tpr[np.argmin(np.abs(fpr - 0.001))]), 6),
        "shadow_train_size": int(shadow_train_x.shape[0]),
        "target_eval_size": int(target_x.shape[0]),
    }


def run_gsa_runtime_mainline(
    workspace: str | Path,
    assets_root: str | Path,
    repo_root: str | Path = "workspaces/white-box/external/GSA",
    resolution: int = 32,
    ddpm_num_steps: int = 20,
    sampling_frequency: int = 2,
    attack_method: int = 1,
    prediction_type: str = "epsilon",
    max_samples: int | None = None,
    device: str = "auto",
    provenance_status: str = "workspace-verified",
) -> dict[str, Any]:
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    asset_probe = probe_gsa_assets(assets_root=assets_root, repo_root=repo_root)
    if asset_probe["status"] != "ready":
        result = {
            "status": "blocked",
            "track": "white-box",
            "method": "gsa",
            "paper": "WhiteBox_GSA_PoPETS2025",
            "mode": "runtime-mainline",
            "workspace": str(workspace_path),
            "workspace_name": workspace_path.name,
            "contract_stage": "target",
            "asset_grade": "real-asset-closed-loop",
            "provenance_status": provenance_status,
            "evidence_level": "runtime-mainline",
            "checks": {
                "asset_probe_ready": False,
            },
            "asset_probe": asset_probe,
            "artifact_paths": {
                "summary": str(workspace_path / "summary.json"),
            },
            "notes": [
                "GSA runtime mainline requires dataset buckets, manifest files, and checkpoint-* directories.",
            ],
        }
        (workspace_path / "summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return result

    repo_path = Path(repo_root).resolve()
    requested_device = device.lower()
    resolved_device = _resolve_runtime_device(device)

    gradients_root = workspace_path / "gradients"
    gradients_root.mkdir(parents=True, exist_ok=True)
    shadow_specs = asset_probe["paths"]["shadow_specs"]
    jobs = {
        "target_member": {
            "dataset_dir": asset_probe["paths"]["target_member_dataset"],
            "model_dir": str(Path(asset_probe["paths"]["target_checkpoint"]).parent),
            "output_name": gradients_root / "target_member-gradients.pt",
        },
        "target_non_member": {
            "dataset_dir": asset_probe["paths"]["target_nonmember_dataset"],
            "model_dir": str(Path(asset_probe["paths"]["target_checkpoint"]).parent),
            "output_name": gradients_root / "target_non_member-gradients.pt",
        },
    }
    for spec in shadow_specs:
        job_prefix = spec["name"].replace("-", "_")
        jobs[f"{job_prefix}_member"] = {
            "dataset_dir": spec["member_dataset"],
            "model_dir": spec["checkpoint_dir"],
            "output_name": gradients_root / f"{spec['name']}_member-gradients.pt",
        }
        jobs[f"{job_prefix}_non_member"] = {
            "dataset_dir": spec["nonmember_dataset"],
            "model_dir": spec["checkpoint_dir"],
            "output_name": gradients_root / f"{spec['name']}_non_member-gradients.pt",
        }

    command_results: dict[str, Any] = {}
    all_gradients_ready = True
    for job_name, spec in jobs.items():
        command = [
            sys.executable,
            str((repo_path / "DDPM" / "gen_l2_gradients_DDPM.py").resolve()),
            "--train_data_dir",
            str(Path(spec["dataset_dir"]).resolve()),
            "--model_dir",
            str(Path(spec["model_dir"]).resolve()),
            "--resume_from_checkpoint",
            "latest",
            "--resolution",
            str(int(resolution)),
            "--ddpm_num_steps",
            str(int(ddpm_num_steps)),
            "--sampling_frequency",
            str(int(sampling_frequency)),
            "--attack_method",
            str(int(attack_method)),
            "--prediction_type",
            prediction_type,
            "--output_name",
            str(Path(spec["output_name"]).resolve()),
        ]
        command_env = dict(**os.environ)
        if resolved_device == "cpu":
            command_env["CUDA_VISIBLE_DEVICES"] = "-1"
        command_result = _run_subprocess(command, cwd=repo_path, env=command_env)
        command_results[job_name] = command_result
        if command_result["returncode"] != 0 or not Path(spec["output_name"]).exists():
            all_gradients_ready = False

    if all_gradients_ready:
        shadow_member_paths = [
            jobs[f"{spec['name'].replace('-', '_')}_member"]["output_name"] for spec in shadow_specs
        ]
        shadow_nonmember_paths = [
            jobs[f"{spec['name'].replace('-', '_')}_non_member"]["output_name"] for spec in shadow_specs
        ]
        metrics = _evaluate_gsa_closed_loop(
            target_member_path=jobs["target_member"]["output_name"],
            target_nonmember_path=jobs["target_non_member"]["output_name"],
            shadow_member_paths=shadow_member_paths,
            shadow_nonmember_paths=shadow_nonmember_paths,
            max_samples=max_samples,
        )
        attack_output = workspace_path / "attack-output.txt"
        attack_output.write_text(
            "\n".join(
                [
                    "Target Attack Results -------------------------------",
                    f"AUC: {metrics['auc']}",
                    f"ASR: {metrics['asr']}",
                    f"TPR@1%FPR: {metrics['tpr_at_1pct_fpr']}",
                    f"TPR@0.1%FPR: {metrics['tpr_at_0_1pct_fpr']}",
                ]
            ),
            encoding="utf-8",
        )
    else:
        metrics = {}
        attack_output = workspace_path / "attack-output.txt"

    checks = {
        "asset_probe_ready": True,
        "target_member_gradients": Path(jobs["target_member"]["output_name"]).exists(),
        "target_nonmember_gradients": Path(jobs["target_non_member"]["output_name"]).exists(),
        "shadow_gradients": all(
            Path(jobs[f"{spec['name'].replace('-', '_')}_member"]["output_name"]).exists()
            and Path(jobs[f"{spec['name'].replace('-', '_')}_non_member"]["output_name"]).exists()
            for spec in shadow_specs
        ),
        "closed_loop_metrics_ready": bool(metrics),
    }
    result = {
        "status": "ready" if all(checks.values()) else "error",
        "track": "white-box",
        "method": "gsa",
        "paper": "WhiteBox_GSA_PoPETS2025",
        "mode": "runtime-mainline",
        "workspace": str(workspace_path),
        "workspace_name": workspace_path.name,
        "device": resolved_device,
        "contract_stage": "target",
        "asset_grade": "real-asset-closed-loop",
        "provenance_status": provenance_status,
        "evidence_level": "runtime-mainline",
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
            "attack_output": str(attack_output),
            "target_member_gradients": str(jobs["target_member"]["output_name"]),
            "target_nonmember_gradients": str(jobs["target_non_member"]["output_name"]),
            "shadow_specs": shadow_specs,
        },
        "checks": checks,
        "runtime": {
            "repo_root": str(repo_path),
            "assets_root": str(Path(assets_root).resolve()),
            "requested_device": requested_device,
            "resolution": int(resolution),
            "ddpm_num_steps": int(ddpm_num_steps),
            "sampling_frequency": int(sampling_frequency),
            "attack_method": int(attack_method),
            "prediction_type": prediction_type,
            "max_samples": int(max_samples) if max_samples is not None else None,
            "shadow_count": len(shadow_specs),
        },
        "commands": command_results,
        "metrics": metrics,
        "notes": [
            "Runtime mainline consumes real dataset buckets and checkpoint-* directories for the current GSA DDPM path.",
            "This is a local paper-aligned baseline, not a claim of full published-metric reproduction.",
        ],
    }
    (workspace_path / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return result


def run_gsa_runtime_intervention_review(
    workspace: str | Path,
    assets_root: str | Path,
    mask_summary: str | Path,
    repo_root: str | Path = "workspaces/white-box/external/GSA",
    resolution: int = 32,
    ddpm_num_steps: int = 20,
    sampling_frequency: int = 2,
    attack_method: int = 1,
    prediction_type: str = "epsilon",
    max_samples: int | None = None,
    extraction_max_samples: int | None = None,
    device: str = "cpu",
    provenance_status: str = "workspace-verified",
) -> dict[str, Any]:
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    asset_probe = probe_gsa_assets(assets_root=assets_root, repo_root=repo_root)
    if asset_probe["status"] != "ready":
        result = {
            "status": "blocked",
            "track": "white-box",
            "method": "gsa",
            "mode": "runtime-intervention-review",
            "workspace": str(workspace_path),
            "workspace_name": workspace_path.name,
            "contract_stage": "target-anchored-fixed-mask-review",
            "asset_grade": "real-asset-closed-loop",
            "provenance_status": provenance_status,
            "evidence_level": "bounded-attack-side-intervention-review",
            "checks": {"asset_probe_ready": False},
            "asset_probe": asset_probe,
            "artifact_paths": {"summary": str(workspace_path / "summary.json")},
        }
        (workspace_path / "summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return result

    frozen_mask = _load_frozen_mask_summary(mask_summary)
    requested = frozen_mask["requested"]
    layer_selector = str(requested.get("layer_selector", "mid_block.attentions.0.to_v"))
    mask = frozen_mask["mask"]
    resolved_device = _resolve_runtime_device(device)
    resolved_extraction_max_samples = (
        int(extraction_max_samples) if extraction_max_samples is not None else int(max_samples) if max_samples is not None else None
    )
    gradients_root = workspace_path / "gradients"
    baseline_root = gradients_root / "baseline"
    intervened_root = gradients_root / "intervened"
    baseline_root.mkdir(parents=True, exist_ok=True)
    intervened_root.mkdir(parents=True, exist_ok=True)

    shadow_specs = asset_probe["paths"]["shadow_specs"]
    jobs = {
        "target_member": {
            "dataset_dir": asset_probe["paths"]["target_member_dataset"],
            "checkpoint_root": str(Path(asset_probe["paths"]["target_checkpoint"]).parent),
            "checkpoint_dir": asset_probe["paths"]["target_checkpoint"],
        },
        "target_non_member": {
            "dataset_dir": asset_probe["paths"]["target_nonmember_dataset"],
            "checkpoint_root": str(Path(asset_probe["paths"]["target_checkpoint"]).parent),
            "checkpoint_dir": asset_probe["paths"]["target_checkpoint"],
        },
    }
    for spec in shadow_specs:
        job_prefix = spec["name"].replace("-", "_")
        latest_checkpoint = _latest_checkpoint_dir(spec["checkpoint_dir"])
        jobs[f"{job_prefix}_member"] = {
            "dataset_dir": spec["member_dataset"],
            "checkpoint_root": spec["checkpoint_dir"],
            "checkpoint_dir": str(latest_checkpoint),
        }
        jobs[f"{job_prefix}_non_member"] = {
            "dataset_dir": spec["nonmember_dataset"],
            "checkpoint_root": spec["checkpoint_dir"],
            "checkpoint_dir": str(latest_checkpoint),
        }

    command_results: dict[str, Any] = {}
    baseline_ready = True
    intervened_ready = True
    for job_name, spec in jobs.items():
        baseline_output = baseline_root / f"{job_name}-gradients.pt"
        intervened_output = intervened_root / f"{job_name}-gradients.pt"
        baseline_result = _extract_gsa_gradients_with_fixed_mask(
            dataset_dir=spec["dataset_dir"],
            checkpoint_root=spec["checkpoint_root"],
            checkpoint_dir=spec["checkpoint_dir"],
            output_path=baseline_output,
            layer_selector=layer_selector,
            channel_indices=None,
            alpha=1.0,
            resolution=resolution,
            ddpm_num_steps=ddpm_num_steps,
            sampling_frequency=sampling_frequency,
            attack_method=attack_method,
            prediction_type=prediction_type,
            device=resolved_device,
            extraction_max_samples=resolved_extraction_max_samples,
        )
        intervened_result = _extract_gsa_gradients_with_fixed_mask(
            dataset_dir=spec["dataset_dir"],
            checkpoint_root=spec["checkpoint_root"],
            checkpoint_dir=spec["checkpoint_dir"],
            output_path=intervened_output,
            layer_selector=layer_selector,
            channel_indices=mask["channel_indices"],
            alpha=mask["alpha"],
            resolution=resolution,
            ddpm_num_steps=ddpm_num_steps,
            sampling_frequency=sampling_frequency,
            attack_method=attack_method,
            prediction_type=prediction_type,
            device=resolved_device,
            extraction_max_samples=resolved_extraction_max_samples,
        )
        command_results[job_name] = {
            "baseline": baseline_result,
            "intervened": intervened_result,
        }
        baseline_ready = baseline_ready and Path(baseline_output).exists()
        intervened_ready = intervened_ready and Path(intervened_output).exists()

    baseline_metrics = _evaluate_gsa_closed_loop(
        target_member_path=baseline_root / "target_member-gradients.pt",
        target_nonmember_path=baseline_root / "target_non_member-gradients.pt",
        shadow_member_paths=[baseline_root / f"{spec['name'].replace('-', '_')}_member-gradients.pt" for spec in shadow_specs],
        shadow_nonmember_paths=[baseline_root / f"{spec['name'].replace('-', '_')}_non_member-gradients.pt" for spec in shadow_specs],
        max_samples=max_samples,
    )
    intervened_metrics = _evaluate_gsa_closed_loop(
        target_member_path=intervened_root / "target_member-gradients.pt",
        target_nonmember_path=intervened_root / "target_non_member-gradients.pt",
        shadow_member_paths=[intervened_root / f"{spec['name'].replace('-', '_')}_member-gradients.pt" for spec in shadow_specs],
        shadow_nonmember_paths=[intervened_root / f"{spec['name'].replace('-', '_')}_non_member-gradients.pt" for spec in shadow_specs],
        max_samples=max_samples,
    )
    metric_deltas = {
        "auc_delta": round(intervened_metrics["auc"] - baseline_metrics["auc"], 6),
        "asr_delta": round(intervened_metrics["asr"] - baseline_metrics["asr"], 6),
        "tpr_at_1pct_fpr_delta": round(
            intervened_metrics["tpr_at_1pct_fpr"] - baseline_metrics["tpr_at_1pct_fpr"], 6
        ),
        "tpr_at_0_1pct_fpr_delta": round(
            intervened_metrics["tpr_at_0_1pct_fpr"] - baseline_metrics["tpr_at_0_1pct_fpr"], 6
        ),
    }
    checks = {
        "asset_probe_ready": True,
        "baseline_gradients_ready": baseline_ready,
        "intervened_gradients_ready": intervened_ready,
        "baseline_metrics_ready": bool(baseline_metrics),
        "intervened_metrics_ready": bool(intervened_metrics),
    }
    result = {
        "status": "ready" if all(checks.values()) else "error",
        "track": "white-box",
        "method": "gsa",
        "mode": "runtime-intervention-review",
        "workspace": str(workspace_path),
        "workspace_name": workspace_path.name,
        "device": resolved_device,
        "contract_stage": "target-anchored-fixed-mask-review",
        "asset_grade": "real-asset-closed-loop",
        "provenance_status": provenance_status,
        "evidence_level": "bounded-attack-side-intervention-review",
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
            "baseline_gradients_root": str(baseline_root),
            "intervened_gradients_root": str(intervened_root),
        },
        "checks": checks,
        "runtime": {
            "repo_root": str(Path(repo_root).resolve()),
            "assets_root": str(Path(assets_root).resolve()),
            "mask_summary": frozen_mask["summary_path"],
            "layer_selector": layer_selector,
            "resolution": int(resolution),
            "ddpm_num_steps": int(ddpm_num_steps),
            "sampling_frequency": int(sampling_frequency),
            "attack_method": int(attack_method),
            "prediction_type": prediction_type,
            "max_samples": int(max_samples) if max_samples is not None else None,
            "extraction_max_samples": resolved_extraction_max_samples,
            "shadow_count": len(shadow_specs),
        },
        "mask": mask,
        "baseline": {
            "metrics": baseline_metrics,
        },
        "intervened": {
            "metrics": intervened_metrics,
        },
        "metric_deltas": metric_deltas,
        "locality_anchor": {
            "requested": requested,
            "metrics": frozen_mask["locality_anchor_metrics"],
        },
        "commands": command_results,
        "notes": [
            "The first intervention review reuses one frozen target-anchored mask across target and shadow gradient extraction.",
            "This review remains bounded and must be read jointly with the locality anchor rather than as attack metrics alone.",
        ],
    }
    (workspace_path / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return result


def export_gsa_loss_score_packet(
    workspace: str | Path,
    assets_root: str | Path,
    repo_root: str | Path = "workspaces/white-box/external/GSA",
    resolution: int = 32,
    ddpm_num_steps: int = 20,
    sampling_frequency: int = 2,
    attack_method: int = 1,
    prediction_type: str = "epsilon",
    extraction_max_samples: int | None = None,
    sample_id_allowlist: list[int] | None = None,
    device: str = "cpu",
    provenance_status: str = "workspace-verified",
) -> dict[str, Any]:
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    asset_probe = probe_gsa_assets(assets_root=assets_root, repo_root=repo_root)
    if asset_probe["status"] != "ready":
        result = {
            "status": "blocked",
            "track": "white-box",
            "method": "gsa",
            "paper": "WhiteBox_GSA_PoPETS2025",
            "mode": "loss-score-export",
            "workspace": str(workspace_path),
            "workspace_name": workspace_path.name,
            "contract_stage": "target",
            "asset_grade": "real-asset-closed-loop",
            "provenance_status": provenance_status,
            "evidence_level": "bounded-loss-score-export",
            "checks": {
                "asset_probe_ready": False,
            },
            "asset_probe": asset_probe,
            "artifact_paths": {
                "summary": str(workspace_path / "summary.json"),
            },
            "notes": [
                "GSA bounded loss-score export requires the same dataset buckets, manifest files, and checkpoint-* directories as the admitted gradient path.",
            ],
        }
        (workspace_path / "summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return result

    scores_root = workspace_path / "loss-scores"
    scores_root.mkdir(parents=True, exist_ok=True)
    shadow_specs = asset_probe["paths"]["shadow_specs"]
    jobs = {
        "target_member": {
            "dataset_dir": asset_probe["paths"]["target_member_dataset"],
            "checkpoint_root": str(Path(asset_probe["paths"]["target_checkpoint"]).parent),
            "checkpoint_dir": asset_probe["paths"]["target_checkpoint"],
            "output_name": scores_root / "target_member-loss-scores.pt",
            "records_name": scores_root / "target_member-loss-scores.jsonl",
        },
        "target_non_member": {
            "dataset_dir": asset_probe["paths"]["target_nonmember_dataset"],
            "checkpoint_root": str(Path(asset_probe["paths"]["target_checkpoint"]).parent),
            "checkpoint_dir": asset_probe["paths"]["target_checkpoint"],
            "output_name": scores_root / "target_non_member-loss-scores.pt",
            "records_name": scores_root / "target_non_member-loss-scores.jsonl",
        },
    }
    for spec in shadow_specs:
        job_prefix = spec["name"].replace("-", "_")
        latest_checkpoint = _latest_checkpoint_dir(spec["checkpoint_dir"])
        jobs[f"{job_prefix}_member"] = {
            "dataset_dir": spec["member_dataset"],
            "checkpoint_root": spec["checkpoint_dir"],
            "checkpoint_dir": str(latest_checkpoint),
            "output_name": scores_root / f"{job_prefix}_member-loss-scores.pt",
            "records_name": scores_root / f"{job_prefix}_member-loss-scores.jsonl",
        }
        jobs[f"{job_prefix}_non_member"] = {
            "dataset_dir": spec["nonmember_dataset"],
            "checkpoint_root": spec["checkpoint_dir"],
            "checkpoint_dir": str(latest_checkpoint),
            "output_name": scores_root / f"{job_prefix}_non_member-loss-scores.pt",
            "records_name": scores_root / f"{job_prefix}_non_member-loss-scores.jsonl",
        }

    export_results: dict[str, Any] = {}
    all_exports_ready = True
    for job_name, spec in jobs.items():
        export_result = _extract_gsa_loss_scores(
            dataset_dir=spec["dataset_dir"],
            checkpoint_root=spec["checkpoint_root"],
            checkpoint_dir=spec["checkpoint_dir"],
            output_path=spec["output_name"],
            records_path=spec["records_name"],
            resolution=resolution,
            ddpm_num_steps=ddpm_num_steps,
            sampling_frequency=sampling_frequency,
            attack_method=attack_method,
            prediction_type=prediction_type,
            device=device,
            extraction_max_samples=extraction_max_samples,
            sample_id_allowlist=sample_id_allowlist,
        )
        export_results[job_name] = export_result
        if export_result["status"] != "ready":
            all_exports_ready = False

    checks = {
        "asset_probe_ready": True,
        "target_member_scores": Path(jobs["target_member"]["output_name"]).exists(),
        "target_nonmember_scores": Path(jobs["target_non_member"]["output_name"]).exists(),
        "shadow_scores": all(
            Path(jobs[f"{spec['name'].replace('-', '_')}_member"]["output_name"]).exists()
            and Path(jobs[f"{spec['name'].replace('-', '_')}_non_member"]["output_name"]).exists()
            for spec in shadow_specs
        ),
    }
    result = {
        "status": "ready" if all(checks.values()) and all_exports_ready else "error",
        "track": "white-box",
        "method": "gsa",
        "paper": "WhiteBox_GSA_PoPETS2025",
        "mode": "loss-score-export",
        "workspace": str(workspace_path),
        "workspace_name": workspace_path.name,
        "device": _resolve_runtime_device(device),
        "contract_stage": "target",
        "asset_grade": "real-asset-closed-loop",
        "provenance_status": provenance_status,
        "evidence_level": "bounded-loss-score-export",
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
            "target_member_scores": str(jobs["target_member"]["output_name"]),
            "target_nonmember_scores": str(jobs["target_non_member"]["output_name"]),
            "shadow_specs": shadow_specs,
        },
        "checks": checks,
        "runtime": {
            "assets_root": str(Path(assets_root).resolve()),
            "resolution": int(resolution),
            "ddpm_num_steps": int(ddpm_num_steps),
            "sampling_frequency": int(sampling_frequency),
            "attack_method": int(attack_method),
            "prediction_type": prediction_type,
            "extraction_max_samples": int(extraction_max_samples) if extraction_max_samples is not None else None,
            "sample_id_allowlist_count": int(len(sample_id_allowlist)) if sample_id_allowlist is not None else None,
        },
        "exports": export_results,
        "notes": [
            "This bounded surface exports per-split loss-score artifacts on the same admitted white-box asset family.",
            "It is intentionally separate from the admitted gradient-centered runtime mainline.",
        ],
    }
    (workspace_path / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return result
