"""Defense-native target-only leakage comparator for DPDM W-1 checkpoints."""

from __future__ import annotations

import gc
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve


def _append_progress(workspace_path: Path, message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    with (workspace_path / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def _release_model(model: torch.nn.Module | None, device: str) -> None:
    if model is None:
        return
    del model
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _ensure_dpdm_imports(dpdm_root: str | Path) -> None:
    root = str(Path(dpdm_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_dpdm_model(
    dpdm_root: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    _ensure_dpdm_imports(dpdm_root)

    from denoiser import EDMDenoiser
    from model.ncsnpp import NCSNpp

    config = OmegaConf.load(str(config_path))
    model = EDMDenoiser(NCSNpp(**config.model.network)).to(device)
    payload = torch.load(str(checkpoint_path), map_location=device)
    state_dict = payload["model"]
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        normalized_key = key[len("module.") :] if key.startswith("module.") else key
        cleaned_state_dict[normalized_key] = value
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()
    return model, OmegaConf.to_container(config, resolve=True)


def _list_images(dataset_dir: str | Path) -> list[Path]:
    root = Path(dataset_dir)
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    )


def _load_image_tensor(image_path: Path, device: str) -> torch.Tensor:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor / 127.5 - 1.0


def _parse_label_from_filename(image_path: Path) -> int:
    stem = image_path.stem
    label_token = stem.split("-", 1)[0]
    return int(label_token)


def _sigma_schedule(p_mean: float, p_std: float, sigma_points: int, device: str) -> torch.Tensor:
    if sigma_points <= 1:
        return torch.exp(torch.tensor([p_mean], dtype=torch.float32, device=device))
    offsets = torch.linspace(-1.0, 1.0, sigma_points, dtype=torch.float32, device=device)
    return torch.exp(p_mean + p_std * offsets)


def _score_sample(
    model: torch.nn.Module,
    image: torch.Tensor,
    label: int,
    sigma_values: torch.Tensor,
    seed: int,
) -> float:
    scores: list[float] = []
    label_tensor = torch.tensor([label], dtype=torch.long, device=image.device)

    for sigma_index, sigma in enumerate(sigma_values):
        generator = torch.Generator(device=image.device)
        generator.manual_seed(seed + sigma_index)
        noise = torch.randn(image.shape, generator=generator, device=image.device)
        sigma_view = sigma.view(1, 1, 1, 1)
        noisy_image = image + sigma_view * noise

        model.zero_grad(set_to_none=True)
        denoised = model(noisy_image, sigma_view, label_tensor)
        loss = torch.nn.functional.mse_loss(denoised, image)
        loss.backward()

        grad_norm = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                grad_norm += float(parameter.grad.norm().item())
        scores.append(grad_norm)
        del noise
        del noisy_image
        del denoised
        del loss

    model.zero_grad(set_to_none=True)
    del label_tensor
    if image.device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(np.mean(scores))


def _score_dataset(
    model: torch.nn.Module,
    dataset_dir: str | Path,
    sigma_values: torch.Tensor,
    device: str,
    max_samples: int | None,
    seed_base: int,
) -> list[float]:
    image_paths = _list_images(dataset_dir)
    if max_samples is not None:
        image_paths = image_paths[:max_samples]

    scores: list[float] = []
    for index, image_path in enumerate(image_paths):
        image_tensor = _load_image_tensor(image_path, device=device)
        label = _parse_label_from_filename(image_path)
        score = _score_sample(
            model=model,
            image=image_tensor,
            label=label,
            sigma_values=sigma_values,
            seed=seed_base + index,
        )
        scores.append(score)
        del image_tensor
        if device == "cuda" and torch.cuda.is_available() and (index + 1) % 16 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    return scores


def _metric_bundle(member_scores: list[float], nonmember_scores: list[float]) -> dict[str, Any]:
    y_true = np.concatenate(
        [
            np.ones(len(member_scores), dtype=np.int64),
            np.zeros(len(nonmember_scores), dtype=np.int64),
        ]
    )
    raw_scores = np.concatenate([np.asarray(member_scores), np.asarray(nonmember_scores)])
    member_mean = float(np.mean(member_scores))
    nonmember_mean = float(np.mean(nonmember_scores))
    oriented_scores = raw_scores if member_mean >= nonmember_mean else -raw_scores
    auc = float(roc_auc_score(y_true, oriented_scores))
    fpr, tpr, thresholds = roc_curve(y_true, oriented_scores)

    best_index = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[best_index])
    predictions = (oriented_scores >= threshold).astype(np.int64)
    asr = float(np.mean(predictions == y_true))

    return {
        "auc": round(auc, 6),
        "asr": round(asr, 6),
        "tpr_at_1pct_fpr": round(float(tpr[np.argmin(np.abs(fpr - 0.01))]), 6),
        "tpr_at_0_1pct_fpr": round(float(tpr[np.argmin(np.abs(fpr - 0.001))]), 6),
        "member_mean_score": round(member_mean, 6),
        "nonmember_mean_score": round(nonmember_mean, 6),
        "score_direction": "member-higher" if member_mean >= nonmember_mean else "member-lower",
        "optimal_threshold": threshold,
        "member_eval_size": len(member_scores),
        "nonmember_eval_size": len(nonmember_scores),
    }


def _shadow_target_metric_bundle(
    shadow_member_scores: list[float],
    shadow_nonmember_scores: list[float],
    target_member_scores: list[float],
    target_nonmember_scores: list[float],
) -> dict[str, Any]:
    shadow_x = np.concatenate(
        [np.asarray(shadow_member_scores), np.asarray(shadow_nonmember_scores)]
    ).reshape(-1, 1)
    shadow_y = np.concatenate(
        [
            np.ones(len(shadow_member_scores), dtype=np.int64),
            np.zeros(len(shadow_nonmember_scores), dtype=np.int64),
        ]
    )

    target_x = np.concatenate(
        [np.asarray(target_member_scores), np.asarray(target_nonmember_scores)]
    ).reshape(-1, 1)
    target_y = np.concatenate(
        [
            np.ones(len(target_member_scores), dtype=np.int64),
            np.zeros(len(target_nonmember_scores), dtype=np.int64),
        ]
    )

    classifier = LogisticRegression(random_state=0, max_iter=1000)
    classifier.fit(shadow_x, shadow_y)
    target_prob = classifier.predict_proba(target_x)[:, 1]
    target_pred = classifier.predict(target_x)

    auc = float(roc_auc_score(target_y, target_prob))
    fpr, tpr, _ = roc_curve(target_y, target_prob)
    asr = float(np.mean(target_pred == target_y))

    return {
        "auc": round(auc, 6),
        "asr": round(asr, 6),
        "tpr_at_1pct_fpr": round(float(tpr[np.argmin(np.abs(fpr - 0.01))]), 6),
        "tpr_at_0_1pct_fpr": round(float(tpr[np.argmin(np.abs(fpr - 0.001))]), 6),
        "shadow_member_mean_score": round(float(np.mean(shadow_member_scores)), 6),
        "shadow_nonmember_mean_score": round(float(np.mean(shadow_nonmember_scores)), 6),
        "target_member_mean_score": round(float(np.mean(target_member_scores)), 6),
        "target_nonmember_mean_score": round(float(np.mean(target_nonmember_scores)), 6),
        "shadow_train_size": len(shadow_member_scores) + len(shadow_nonmember_scores),
        "target_eval_size": len(target_member_scores) + len(target_nonmember_scores),
        "classifier": "logistic-regression-1d",
    }


def _multi_shadow_target_metric_bundle(
    shadow_member_score_groups: list[list[float]],
    shadow_nonmember_score_groups: list[list[float]],
    target_member_scores: list[float],
    target_nonmember_scores: list[float],
) -> dict[str, Any]:
    flat_shadow_member = [score for group in shadow_member_score_groups for score in group]
    flat_shadow_nonmember = [score for group in shadow_nonmember_score_groups for score in group]
    return _shadow_target_metric_bundle(
        shadow_member_scores=flat_shadow_member,
        shadow_nonmember_scores=flat_shadow_nonmember,
        target_member_scores=target_member_scores,
        target_nonmember_scores=target_nonmember_scores,
    )


def run_dpdm_w1_target_only_comparator(
    workspace: str | Path,
    checkpoint_path: str | Path,
    member_dataset_dir: str | Path,
    nonmember_dataset_dir: str | Path,
    dpdm_root: str | Path = "external/DPDM",
    config_path: str | Path = "external/DPDM/configs/cifar10_32/train_eps_10.0.yaml",
    device: str = "cuda",
    sigma_points: int = 8,
    max_samples: int | None = 128,
    provenance_status: str = "workspace-verified",
) -> dict[str, Any]:
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    resolved_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    model, config = _load_dpdm_model(
        dpdm_root=dpdm_root,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=resolved_device,
    )

    sigma_values = _sigma_schedule(
        p_mean=float(config["loss"]["p_mean"]),
        p_std=float(config["loss"]["p_std"]),
        sigma_points=sigma_points,
        device=resolved_device,
    )

    member_scores = _score_dataset(
        model=model,
        dataset_dir=member_dataset_dir,
        sigma_values=sigma_values,
        device=resolved_device,
        max_samples=max_samples,
        seed_base=0,
    )
    nonmember_scores = _score_dataset(
        model=model,
        dataset_dir=nonmember_dataset_dir,
        sigma_values=sigma_values,
        device=resolved_device,
        max_samples=max_samples,
        seed_base=10_000,
    )

    metrics = _metric_bundle(member_scores, nonmember_scores)
    score_payload = {
        "member_scores": member_scores,
        "nonmember_scores": nonmember_scores,
    }
    (workspace_path / "scores.json").write_text(
        json.dumps(score_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    attack_output_lines = [
        "DPDM W-1 Target-Only Comparator -------------------------------",
        f"AUC: {metrics['auc']}",
        f"ASR: {metrics['asr']}",
        f"TPR@1%FPR: {metrics['tpr_at_1pct_fpr']}",
        f"TPR@0.1%FPR: {metrics['tpr_at_0_1pct_fpr']}",
        f"member_mean_score: {metrics['member_mean_score']}",
        f"nonmember_mean_score: {metrics['nonmember_mean_score']}",
    ]
    (workspace_path / "attack-output.txt").write_text(
        "\n".join(attack_output_lines),
        encoding="utf-8",
    )

    result = {
        "status": "ready",
        "track": "white-box",
        "method": "dpdm-w1-target-only",
        "paper": "DPDM_TMLR2023",
        "mode": "runtime-comparator",
        "workspace": str(workspace_path.resolve()),
        "workspace_name": workspace_path.name,
        "device": resolved_device,
        "contract_stage": "target",
        "asset_grade": "runtime-smoke",
        "provenance_status": provenance_status,
        "evidence_level": "runtime-smoke",
        "artifact_paths": {
            "summary": str((workspace_path / "summary.json").resolve()),
            "attack_output": str((workspace_path / "attack-output.txt").resolve()),
            "scores": str((workspace_path / "scores.json").resolve()),
            "checkpoint": str(Path(checkpoint_path).resolve()),
        },
        "runtime": {
            "dpdm_root": str(Path(dpdm_root).resolve()),
            "config_path": str(Path(config_path).resolve()),
            "member_dataset_dir": str(Path(member_dataset_dir).resolve()),
            "nonmember_dataset_dir": str(Path(nonmember_dataset_dir).resolve()),
            "sigma_points": sigma_points,
            "max_samples": max_samples,
        },
        "metrics": metrics,
        "notes": [
            "This is a defense-native target-only comparator for DPDM W-1 checkpoints.",
            "It is not yet a shadow-trained white-box attack pipeline.",
        ],
    }
    (workspace_path / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return result


def run_dpdm_w1_shadow_comparator(
    workspace: str | Path,
    target_checkpoint_path: str | Path,
    shadow_checkpoint_path: str | Path,
    target_member_dataset_dir: str | Path,
    target_nonmember_dataset_dir: str | Path,
    shadow_member_dataset_dir: str | Path,
    shadow_nonmember_dataset_dir: str | Path,
    dpdm_root: str | Path = "external/DPDM",
    config_path: str | Path = "external/DPDM/configs/cifar10_32/train_eps_10.0.yaml",
    device: str = "cuda",
    sigma_points: int = 8,
    max_samples: int | None = 128,
    provenance_status: str = "workspace-verified",
) -> dict[str, Any]:
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    resolved_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    target_model, config = _load_dpdm_model(
        dpdm_root=dpdm_root,
        config_path=config_path,
        checkpoint_path=target_checkpoint_path,
        device=resolved_device,
    )
    shadow_model, _ = _load_dpdm_model(
        dpdm_root=dpdm_root,
        config_path=config_path,
        checkpoint_path=shadow_checkpoint_path,
        device=resolved_device,
    )

    sigma_values = _sigma_schedule(
        p_mean=float(config["loss"]["p_mean"]),
        p_std=float(config["loss"]["p_std"]),
        sigma_points=sigma_points,
        device=resolved_device,
    )

    shadow_member_scores = _score_dataset(
        model=shadow_model,
        dataset_dir=shadow_member_dataset_dir,
        sigma_values=sigma_values,
        device=resolved_device,
        max_samples=max_samples,
        seed_base=20_000,
    )
    shadow_nonmember_scores = _score_dataset(
        model=shadow_model,
        dataset_dir=shadow_nonmember_dataset_dir,
        sigma_values=sigma_values,
        device=resolved_device,
        max_samples=max_samples,
        seed_base=30_000,
    )
    target_member_scores = _score_dataset(
        model=target_model,
        dataset_dir=target_member_dataset_dir,
        sigma_values=sigma_values,
        device=resolved_device,
        max_samples=max_samples,
        seed_base=0,
    )
    target_nonmember_scores = _score_dataset(
        model=target_model,
        dataset_dir=target_nonmember_dataset_dir,
        sigma_values=sigma_values,
        device=resolved_device,
        max_samples=max_samples,
        seed_base=10_000,
    )

    metrics = _shadow_target_metric_bundle(
        shadow_member_scores=shadow_member_scores,
        shadow_nonmember_scores=shadow_nonmember_scores,
        target_member_scores=target_member_scores,
        target_nonmember_scores=target_nonmember_scores,
    )

    score_payload = {
        "shadow_member_scores": shadow_member_scores,
        "shadow_nonmember_scores": shadow_nonmember_scores,
        "target_member_scores": target_member_scores,
        "target_nonmember_scores": target_nonmember_scores,
    }
    (workspace_path / "scores.json").write_text(
        json.dumps(score_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    attack_output_lines = [
        "DPDM W-1 Shadow Comparator -------------------------------",
        f"AUC: {metrics['auc']}",
        f"ASR: {metrics['asr']}",
        f"TPR@1%FPR: {metrics['tpr_at_1pct_fpr']}",
        f"TPR@0.1%FPR: {metrics['tpr_at_0_1pct_fpr']}",
        f"shadow_member_mean_score: {metrics['shadow_member_mean_score']}",
        f"shadow_nonmember_mean_score: {metrics['shadow_nonmember_mean_score']}",
        f"target_member_mean_score: {metrics['target_member_mean_score']}",
        f"target_nonmember_mean_score: {metrics['target_nonmember_mean_score']}",
    ]
    (workspace_path / "attack-output.txt").write_text(
        "\n".join(attack_output_lines),
        encoding="utf-8",
    )

    result = {
        "status": "ready",
        "track": "white-box",
        "method": "dpdm-w1-shadow-comparator",
        "paper": "DPDM_TMLR2023",
        "mode": "runtime-comparator",
        "workspace": str(workspace_path.resolve()),
        "workspace_name": workspace_path.name,
        "device": resolved_device,
        "contract_stage": "target",
        "asset_grade": "runtime-smoke",
        "provenance_status": provenance_status,
        "evidence_level": "runtime-smoke",
        "artifact_paths": {
            "summary": str((workspace_path / "summary.json").resolve()),
            "attack_output": str((workspace_path / "attack-output.txt").resolve()),
            "scores": str((workspace_path / "scores.json").resolve()),
            "target_checkpoint": str(Path(target_checkpoint_path).resolve()),
            "shadow_checkpoint": str(Path(shadow_checkpoint_path).resolve()),
        },
        "runtime": {
            "dpdm_root": str(Path(dpdm_root).resolve()),
            "config_path": str(Path(config_path).resolve()),
            "target_member_dataset_dir": str(Path(target_member_dataset_dir).resolve()),
            "target_nonmember_dataset_dir": str(Path(target_nonmember_dataset_dir).resolve()),
            "shadow_member_dataset_dir": str(Path(shadow_member_dataset_dir).resolve()),
            "shadow_nonmember_dataset_dir": str(Path(shadow_nonmember_dataset_dir).resolve()),
            "sigma_points": sigma_points,
            "max_samples": max_samples,
        },
        "metrics": metrics,
        "notes": [
            "This is a defended shadow-trained comparator for DPDM W-1 checkpoints.",
            "It remains defense-native and is not a direct reuse of the GSA UNet pipeline.",
        ],
    }
    (workspace_path / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return result


def run_dpdm_w1_multi_shadow_comparator(
    workspace: str | Path,
    target_checkpoint_path: str | Path,
    shadow_checkpoint_paths: list[str | Path],
    target_member_dataset_dir: str | Path,
    target_nonmember_dataset_dir: str | Path,
    shadow_member_dataset_dirs: list[str | Path],
    shadow_nonmember_dataset_dirs: list[str | Path],
    dpdm_root: str | Path = "external/DPDM",
    config_path: str | Path = "external/DPDM/configs/cifar10_32/train_eps_10.0.yaml",
    device: str = "cuda",
    sigma_points: int = 8,
    max_samples: int | None = 128,
    provenance_status: str = "workspace-verified",
) -> dict[str, Any]:
    if not (
        len(shadow_checkpoint_paths)
        == len(shadow_member_dataset_dirs)
        == len(shadow_nonmember_dataset_dirs)
    ):
        raise ValueError("shadow checkpoint and dataset lists must have equal length")

    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    _append_progress(workspace_path, "starting multi-shadow comparator")

    resolved_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    _append_progress(workspace_path, f"resolved_device={resolved_device}")
    target_model, config = _load_dpdm_model(
        dpdm_root=dpdm_root,
        config_path=config_path,
        checkpoint_path=target_checkpoint_path,
        device=resolved_device,
    )
    _append_progress(workspace_path, f"loaded target checkpoint: {Path(target_checkpoint_path).resolve()}")

    sigma_values = _sigma_schedule(
        p_mean=float(config["loss"]["p_mean"]),
        p_std=float(config["loss"]["p_std"]),
        sigma_points=sigma_points,
        device=resolved_device,
    )
    _append_progress(workspace_path, f"built sigma schedule with {sigma_points} points")

    shadow_member_score_groups: list[list[float]] = []
    shadow_nonmember_score_groups: list[list[float]] = []
    for index, checkpoint_path in enumerate(shadow_checkpoint_paths):
        _append_progress(
            workspace_path,
            f"loading shadow checkpoint {index + 1}/{len(shadow_checkpoint_paths)}: {Path(checkpoint_path).resolve()}",
        )
        shadow_model, _ = _load_dpdm_model(
            dpdm_root=dpdm_root,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=resolved_device,
        )
        _append_progress(workspace_path, f"scoring shadow member set {index + 1}")
        shadow_member_score_groups.append(
            _score_dataset(
                model=shadow_model,
                dataset_dir=shadow_member_dataset_dirs[index],
                sigma_values=sigma_values,
                device=resolved_device,
                max_samples=max_samples,
                seed_base=20_000 * (index + 1),
            )
        )
        _append_progress(workspace_path, f"scoring shadow nonmember set {index + 1}")
        shadow_nonmember_score_groups.append(
            _score_dataset(
                model=shadow_model,
                dataset_dir=shadow_nonmember_dataset_dirs[index],
                sigma_values=sigma_values,
                device=resolved_device,
                max_samples=max_samples,
                seed_base=30_000 * (index + 1),
            )
        )
        _append_progress(workspace_path, f"completed shadow scoring {index + 1}")
        _release_model(shadow_model, resolved_device)

    _append_progress(workspace_path, "scoring target member set")
    target_member_scores = _score_dataset(
        model=target_model,
        dataset_dir=target_member_dataset_dir,
        sigma_values=sigma_values,
        device=resolved_device,
        max_samples=max_samples,
        seed_base=0,
    )
    _append_progress(workspace_path, "scoring target nonmember set")
    target_nonmember_scores = _score_dataset(
        model=target_model,
        dataset_dir=target_nonmember_dataset_dir,
        sigma_values=sigma_values,
        device=resolved_device,
        max_samples=max_samples,
        seed_base=10_000,
    )
    _append_progress(workspace_path, "completed target scoring")
    _release_model(target_model, resolved_device)

    metrics = _multi_shadow_target_metric_bundle(
        shadow_member_score_groups=shadow_member_score_groups,
        shadow_nonmember_score_groups=shadow_nonmember_score_groups,
        target_member_scores=target_member_scores,
        target_nonmember_scores=target_nonmember_scores,
    )
    _append_progress(workspace_path, "computed metrics")

    score_payload = {
        "shadow_member_scores": shadow_member_score_groups,
        "shadow_nonmember_scores": shadow_nonmember_score_groups,
        "target_member_scores": target_member_scores,
        "target_nonmember_scores": target_nonmember_scores,
    }
    (workspace_path / "scores.json").write_text(
        json.dumps(score_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    attack_output_lines = [
        "DPDM W-1 Multi-Shadow Comparator -------------------------------",
        f"AUC: {metrics['auc']}",
        f"ASR: {metrics['asr']}",
        f"TPR@1%FPR: {metrics['tpr_at_1pct_fpr']}",
        f"TPR@0.1%FPR: {metrics['tpr_at_0_1pct_fpr']}",
    ]
    (workspace_path / "attack-output.txt").write_text(
        "\n".join(attack_output_lines),
        encoding="utf-8",
    )

    result = {
        "status": "ready",
        "track": "white-box",
        "method": "dpdm-w1-multi-shadow-comparator",
        "paper": "DPDM_TMLR2023",
        "mode": "runtime-comparator",
        "workspace": str(workspace_path.resolve()),
        "workspace_name": workspace_path.name,
        "device": resolved_device,
        "contract_stage": "target",
        "asset_grade": "runtime-smoke",
        "provenance_status": provenance_status,
        "evidence_level": "runtime-smoke",
        "artifact_paths": {
            "summary": str((workspace_path / "summary.json").resolve()),
            "attack_output": str((workspace_path / "attack-output.txt").resolve()),
            "scores": str((workspace_path / "scores.json").resolve()),
            "target_checkpoint": str(Path(target_checkpoint_path).resolve()),
            "shadow_checkpoints": [str(Path(path).resolve()) for path in shadow_checkpoint_paths],
        },
        "runtime": {
            "dpdm_root": str(Path(dpdm_root).resolve()),
            "config_path": str(Path(config_path).resolve()),
            "target_member_dataset_dir": str(Path(target_member_dataset_dir).resolve()),
            "target_nonmember_dataset_dir": str(Path(target_nonmember_dataset_dir).resolve()),
            "shadow_member_dataset_dirs": [str(Path(path).resolve()) for path in shadow_member_dataset_dirs],
            "shadow_nonmember_dataset_dirs": [str(Path(path).resolve()) for path in shadow_nonmember_dataset_dirs],
            "sigma_points": sigma_points,
            "max_samples": max_samples,
            "shadow_count": len(shadow_checkpoint_paths),
        },
        "metrics": metrics,
        "notes": [
            "This is a defended multi-shadow comparator for DPDM W-1 checkpoints.",
            "It remains defense-native and is not a direct reuse of the GSA UNet pipeline.",
        ],
    }
    (workspace_path / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    _append_progress(workspace_path, "wrote summary artifacts")
    return result
