"""Asset probe helpers for 04-H2 privacy-aware adapter."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from safetensors.torch import load_file, save_file


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_H2_SHADOW_REFERENCE_SUMMARY = (
    "workspaces/white-box/runs/gsa-loss-score-export-bounded-actual-20260418-r1/summary.json"
)
CHECKPOINT_FILE_CANDIDATES = (
    "model.safetensors",
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.bin",
    "pytorch_model.bin",
    "checkpoint.pt",
)
_CHECKPOINT_STEP_PATTERN = re.compile(r"checkpoint-(\d+)$")


def _round6(value: float) -> float:
    return round(float(value), 6)


def _list_image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _resolve_checkpoint_dir(
    checkpoint_root: str | Path,
    checkpoint_dir: str | Path | None = None,
) -> tuple[Path, str, bool]:
    root = Path(checkpoint_root)
    if checkpoint_dir is not None:
        resolved = Path(checkpoint_dir)
        return resolved, "explicit-checkpoint-dir", resolved.exists()

    if not root.exists():
        return root, "checkpoint-root-missing", False

    for candidate_name in CHECKPOINT_FILE_CANDIDATES:
        direct_candidate = root / candidate_name
        if direct_candidate.exists():
            return root, "checkpoint-root-direct", True

    candidate_dirs = [path for path in root.iterdir() if path.is_dir()]
    if not candidate_dirs:
        return root, "checkpoint-root-empty", False

    def sort_key(path: Path) -> tuple[int, str]:
        match = _CHECKPOINT_STEP_PATTERN.search(path.name)
        step = int(match.group(1)) if match else -1
        return (step, path.name)

    resolved = sorted(candidate_dirs, key=sort_key)[-1]
    return resolved, "checkpoint-root-latest", True


def _find_checkpoint_model_file(checkpoint_dir: Path) -> Path | None:
    for candidate_name in CHECKPOINT_FILE_CANDIDATES:
        candidate = checkpoint_dir / candidate_name
        if candidate.exists():
            return candidate
    return None


def _load_checkpoint_meta(checkpoint_dir: Path) -> dict[str, Any] | None:
    meta_path = checkpoint_dir / "checkpoint_meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"error": f"invalid checkpoint_meta.json at {meta_path}"}


def _profile_image_root(
    dataset_dir: str | Path,
    *,
    max_layout_checks: int | None = None,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    profile: dict[str, Any] = {
        "path": str(root),
        "exists": root.exists(),
        "image_count": 0,
        "layout_scan_count": 0,
        "layout_scan_complete": False,
        "layout_consistent": False,
        "first_image": None,
        "image_shape": None,
        "image_mode": None,
        "scanned_shapes": [],
        "error": None,
    }
    if not root.exists():
        return profile

    image_paths = _list_image_files(root)
    profile["image_count"] = len(image_paths)
    if not image_paths:
        return profile

    scan_paths = image_paths if max_layout_checks is None else image_paths[:max_layout_checks]
    scanned_shapes: list[list[int]] = []
    scanned_modes: list[str] = []
    first_shape: list[int] | None = None
    first_mode: str | None = None
    for image_path in scan_paths:
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                channel_count = len(image.getbands())
                current_shape = [int(height), int(width), int(channel_count)]
                current_mode = str(image.mode)
        except Exception as exc:  # pragma: no cover - rare corrupt-image branch
            profile["error"] = f"{type(exc).__name__}: {exc}"
            profile["layout_scan_count"] = len(scanned_shapes)
            return profile

        if first_shape is None:
            first_shape = current_shape
            first_mode = current_mode
            profile["first_image"] = str(image_path)

        scanned_shapes.append(current_shape)
        scanned_modes.append(current_mode)

    unique_shapes = sorted({tuple(shape) for shape in scanned_shapes})
    unique_modes = sorted(set(scanned_modes))
    profile.update(
        {
            "layout_scan_count": len(scan_paths),
            "layout_scan_complete": len(scan_paths) == len(image_paths),
            "layout_consistent": len(unique_shapes) == 1 and len(unique_modes) == 1,
            "image_shape": first_shape,
            "image_mode": first_mode,
            "scanned_shapes": [list(shape) for shape in unique_shapes],
        }
    )
    return profile


def _build_blocker_reason(checks: dict[str, bool]) -> str | None:
    missing_asset_keys = {
        "checkpoint_root",
        "checkpoint_dir",
        "member_dataset_dir",
        "nonmember_dataset_dir",
        "member_has_images",
        "nonmember_has_images",
    }
    if not checks["checkpoint_root"] or not checks["member_dataset_dir"] or not checks["nonmember_dataset_dir"]:
        return "missing-required-assets"
    if not checks["checkpoint_dir"]:
        return "missing-required-assets"
    if not checks["checkpoint_model_file"]:
        return "checkpoint-identity-unresolved"
    if any(not checks[key] for key in ("member_has_images", "nonmember_has_images")):
        return "missing-required-assets"
    if any(
        not checks[key]
        for key in (
            "member_layout_consistent",
            "nonmember_layout_consistent",
            "same_image_shape",
            "same_image_mode",
        )
    ):
        return "image-layout-mismatch"
    if not checks["packet_non_empty"]:
        return "empty-packet"
    if not checks["packet_within_cap"]:
        return "packet-exceeds-cap"
    if any(not checks[key] for key in checks if key not in missing_asset_keys):
        return "blocked"
    return None


def probe_h2_assets(
    *,
    checkpoint_root: str | Path,
    member_dataset_dir: str | Path,
    nonmember_dataset_dir: str | Path,
    checkpoint_dir: str | Path | None = None,
    packet_cap: int = 1000,
    max_layout_checks: int | None = None,
    provenance_status: str = "workspace-verified",
) -> dict[str, Any]:
    if int(packet_cap) <= 0:
        raise ValueError(f"packet_cap must be positive, got {packet_cap}")

    normalized_max_layout_checks = (
        None if max_layout_checks is None or int(max_layout_checks) <= 0 else int(max_layout_checks)
    )

    checkpoint_root_path = Path(checkpoint_root)
    resolved_checkpoint_dir, source_kind, checkpoint_dir_exists = _resolve_checkpoint_dir(
        checkpoint_root=checkpoint_root,
        checkpoint_dir=checkpoint_dir,
    )
    checkpoint_model_file = (
        _find_checkpoint_model_file(resolved_checkpoint_dir) if checkpoint_dir_exists else None
    )
    checkpoint_meta = (
        _load_checkpoint_meta(resolved_checkpoint_dir) if checkpoint_dir_exists else None
    )

    member_profile = _profile_image_root(
        member_dataset_dir,
        max_layout_checks=normalized_max_layout_checks,
    )
    nonmember_profile = _profile_image_root(
        nonmember_dataset_dir,
        max_layout_checks=normalized_max_layout_checks,
    )

    effective_packet_size = min(
        int(member_profile["image_count"]),
        int(nonmember_profile["image_count"]),
    )
    compatibility = {
        "layout_compatible": bool(
            member_profile["layout_consistent"]
            and nonmember_profile["layout_consistent"]
            and member_profile["image_shape"] == nonmember_profile["image_shape"]
            and member_profile["image_mode"] == nonmember_profile["image_mode"]
        ),
        "member_image_shape": member_profile["image_shape"],
        "nonmember_image_shape": nonmember_profile["image_shape"],
        "member_image_mode": member_profile["image_mode"],
        "nonmember_image_mode": nonmember_profile["image_mode"],
        "member_layout_scan_count": int(member_profile["layout_scan_count"]),
        "nonmember_layout_scan_count": int(nonmember_profile["layout_scan_count"]),
        "member_layout_scan_complete": bool(member_profile["layout_scan_complete"]),
        "nonmember_layout_scan_complete": bool(nonmember_profile["layout_scan_complete"]),
        "packet_within_cap": bool(effective_packet_size <= int(packet_cap)),
    }

    checks = {
        "checkpoint_root": checkpoint_root_path.exists(),
        "checkpoint_dir": bool(checkpoint_dir_exists),
        "checkpoint_model_file": checkpoint_model_file is not None,
        "member_dataset_dir": bool(member_profile["exists"]),
        "nonmember_dataset_dir": bool(nonmember_profile["exists"]),
        "member_has_images": int(member_profile["image_count"]) > 0,
        "nonmember_has_images": int(nonmember_profile["image_count"]) > 0,
        "member_layout_consistent": bool(member_profile["layout_consistent"]),
        "nonmember_layout_consistent": bool(nonmember_profile["layout_consistent"]),
        "same_image_shape": member_profile["image_shape"] == nonmember_profile["image_shape"],
        "same_image_mode": member_profile["image_mode"] == nonmember_profile["image_mode"],
        "packet_non_empty": effective_packet_size > 0,
        "packet_within_cap": bool(effective_packet_size <= int(packet_cap)),
    }
    missing_keys = [key for key, is_ready in checks.items() if not is_ready]
    blocker_reason = _build_blocker_reason(checks)
    status = "ready" if blocker_reason is None else "blocked"

    packet_identity = {
        "member_count": int(member_profile["image_count"]),
        "nonmember_count": int(nonmember_profile["image_count"]),
        "effective_packet_size": int(effective_packet_size),
        "packet_cap": int(packet_cap),
        "identity_tag": (
            f"member={int(member_profile['image_count'])}"
            f"|nonmember={int(nonmember_profile['image_count'])}"
            f"|shape={member_profile['image_shape']}"
        ),
    }

    return {
        "status": status,
        "track": "defense",
        "method": "privacy-aware-adapter",
        "paper": "SMP-LoRA_AAAI2025",
        "mode": "asset-probe",
        "contract_stage": "asset-probe",
        "gpu_release": "none",
        "admitted_change": "none",
        "provenance_status": provenance_status,
        "blocker_reason": blocker_reason,
        "checks": checks,
        "missing_keys": missing_keys,
        "checkpoint": {
            "checkpoint_root": str(checkpoint_root_path),
            "resolved_checkpoint_dir": str(resolved_checkpoint_dir),
            "source_kind": source_kind,
            "model_file": str(checkpoint_model_file) if checkpoint_model_file is not None else None,
            "checkpoint_meta": checkpoint_meta,
        },
        "member_dataset": member_profile,
        "nonmember_dataset": nonmember_profile,
        "compatibility": compatibility,
        "packet": packet_identity,
        "runtime": {
            "packet_cap": int(packet_cap),
            "max_layout_checks": normalized_max_layout_checks,
        },
        "notes": [
            "This probe only freezes asset readiness for 04-H2 and does not claim prepare/run/review readiness.",
            "Layout compatibility is verified over the scanned image subset; counts are computed over the full directory tree.",
        ],
    }


def prepare_h2_contract(
    *,
    workspace: str | Path,
    checkpoint_root: str | Path,
    member_dataset_dir: str | Path,
    nonmember_dataset_dir: str | Path,
    checkpoint_dir: str | Path | None = None,
    packet_cap: int = 1000,
    max_layout_checks: int | None = None,
    rank: int = 4,
    alpha: float = 1.0,
    lambda_coeff: float = 0.5,
    delta: float = 1e-4,
    lora_lr: float = 1e-4,
    proxy_lr: float = 1e-3,
    optimizer: str = "adam",
    sgd_momentum: float = 0.9,
    proxy_hidden_dim: int = 256,
    proxy_steps: int = 5,
    num_epochs: int = 10,
    batch_size: int = 8,
    num_workers: int = 0,
    method: str = "smp",
    device: str = "cpu",
    provenance_status: str = "workspace-verified",
) -> dict[str, Any]:
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    probe = probe_h2_assets(
        checkpoint_root=checkpoint_root,
        checkpoint_dir=checkpoint_dir,
        member_dataset_dir=member_dataset_dir,
        nonmember_dataset_dir=nonmember_dataset_dir,
        packet_cap=packet_cap,
        max_layout_checks=max_layout_checks,
        provenance_status=provenance_status,
    )
    probe_summary_path = workspace_path / "probe-summary.json"
    probe_summary_path.write_text(
        json.dumps(probe, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    if probe["status"] != "ready":
        blocked_summary = {
            "status": "blocked",
            "track": "defense",
            "method": "privacy-aware-adapter",
            "mode": "prepare-contract",
            "contract_stage": "prepare-contract",
            "gpu_release": "none",
            "admitted_change": "none",
            "provenance_status": provenance_status,
            "blocker_reason": "probe-not-ready",
            "workspace": str(workspace_path),
            "artifact_paths": {
                "summary": str(workspace_path / "summary.json"),
                "probe_summary": str(probe_summary_path),
            },
            "probe": probe,
        }
        summary_path = workspace_path / "summary.json"
        summary_path.write_text(
            json.dumps(blocked_summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return blocked_summary

    manifest = {
        "status": "ready",
        "track": "defense",
        "method": "privacy-aware-adapter",
        "mode": "prepare-contract",
        "contract_stage": "prepare-contract",
        "gpu_release": "none",
        "admitted_change": "none",
        "provenance_status": provenance_status,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "baseline_checkpoint": probe["checkpoint"],
        "packet": probe["packet"],
        "dataset_contract": {
            "member_dataset": {
                "path": probe["member_dataset"]["path"],
                "image_count": probe["member_dataset"]["image_count"],
                "image_shape": probe["member_dataset"]["image_shape"],
                "image_mode": probe["member_dataset"]["image_mode"],
            },
            "nonmember_dataset": {
                "path": probe["nonmember_dataset"]["path"],
                "image_count": probe["nonmember_dataset"]["image_count"],
                "image_shape": probe["nonmember_dataset"]["image_shape"],
                "image_mode": probe["nonmember_dataset"]["image_mode"],
            },
        },
        "runtime": {
            "rank": int(rank),
            "alpha": float(alpha),
            "lambda_coeff": float(lambda_coeff),
            "delta": float(delta),
            "lora_lr": float(lora_lr),
            "proxy_lr": float(proxy_lr),
            "optimizer": str(optimizer),
            "sgd_momentum": float(sgd_momentum),
            "proxy_hidden_dim": int(proxy_hidden_dim),
            "proxy_steps": int(proxy_steps),
            "num_epochs": int(num_epochs),
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "method": str(method),
            "device": str(device),
        },
    }
    manifest_path = workspace_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    summary = {
        "status": "ready",
        "track": "defense",
        "method": "privacy-aware-adapter",
        "mode": "prepare-contract",
        "contract_stage": "prepare-contract",
        "gpu_release": "none",
        "admitted_change": "none",
        "provenance_status": provenance_status,
        "workspace": str(workspace_path),
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
            "manifest": str(manifest_path),
            "probe_summary": str(probe_summary_path),
        },
        "baseline_checkpoint": probe["checkpoint"],
        "packet": probe["packet"],
        "runtime": manifest["runtime"],
        "notes": [
            "This step freezes the first honest 04-H2 workspace contract but does not execute training.",
            "A subsequent run-h2-defense-pilot stage is still required before any attack-side review or GPU release discussion.",
        ],
    }
    summary_path = workspace_path / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary


def _load_prepare_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.exists():
        return {
            "status": "blocked",
            "blocker_reason": "prepare-manifest-missing",
            "manifest_path": str(path),
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "status": "blocked",
            "blocker_reason": "prepare-manifest-invalid",
            "manifest_path": str(path),
            "error": str(exc),
        }
    if payload.get("status") != "ready" or payload.get("mode") != "prepare-contract":
        return {
            "status": "blocked",
            "blocker_reason": "prepare-manifest-not-ready",
            "manifest_path": str(path),
            "manifest": payload,
        }
    return payload


def _stage_dataset_subset(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    source_root = Path(dataset_dir)
    staged_root = Path(output_dir)
    staged_root.mkdir(parents=True, exist_ok=True)
    image_paths = _list_image_files(source_root)
    if sample_limit is not None:
        image_paths = image_paths[: int(sample_limit)]
    staged_paths: list[str] = []
    for index, source_path in enumerate(image_paths):
        target_path = staged_root / f"{index:04d}-{source_path.name}"
        shutil.copy2(source_path, target_path)
        staged_paths.append(str(target_path))
    return {
        "source_root": str(source_root),
        "staged_root": str(staged_root),
        "count": int(len(staged_paths)),
        "staged_paths": staged_paths,
    }


def _run_h2_training_command(command: list[str], cwd: str | Path) -> tuple[int, str, str]:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    return int(completed.returncode), completed.stdout, completed.stderr


def _read_json_file(path: str | Path) -> dict[str, Any] | list[Any] | None:
    target_path = Path(path)
    if not target_path.exists():
        return None
    return json.loads(target_path.read_text(encoding="utf-8"))


def _merge_lora_linear_weight(module: Any) -> None:
    delta_weight = (module.lora_A @ module.lora_B).transpose(0, 1) * float(module.scaling)
    module.original.weight.data.copy_(module.original.weight.data + delta_weight.to(module.original.weight.data.device))


def _merge_lora_modules_in_place(module: Any) -> None:
    from diffaudit.defenses.lora_ddpm import LoRALinear

    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            _merge_lora_linear_weight(child)
            setattr(module, name, child.original)
        elif isinstance(child, torch.nn.ModuleList):
            for index, item in enumerate(list(child)):
                if isinstance(item, LoRALinear):
                    _merge_lora_linear_weight(item)
                    child[index] = item.original
                else:
                    _merge_lora_modules_in_place(item)
        else:
            _merge_lora_modules_in_place(child)


def _materialize_h2_review_checkpoint(
    *,
    baseline_checkpoint_dir: str | Path,
    lora_weights_path: str | Path,
    output_dir: str | Path,
    rank: int,
    alpha: float,
) -> Path:
    from scripts.train_smp_lora import create_ddpm_model
    from diffaudit.defenses.lora_ddpm import inject_lora_into_unet, load_lora_state_dict

    baseline_dir = Path(baseline_checkpoint_dir)
    baseline_weights_path = baseline_dir / "model.safetensors"
    if not baseline_weights_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint weights not found: {baseline_weights_path}")

    model = create_ddpm_model()
    state_dict = load_file(str(baseline_weights_path))
    model.load_state_dict(state_dict, strict=True)
    inject_lora_into_unet(model, rank=int(rank), alpha=float(alpha))
    lora_state = torch.load(Path(lora_weights_path), map_location="cpu", weights_only=False)
    load_lora_state_dict(model, lora_state)
    _merge_lora_modules_in_place(model)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    merged_weights_path = output_path / "model.safetensors"
    save_file(model.state_dict(), str(merged_weights_path))
    return merged_weights_path


def run_h2_defense_pilot(
    *,
    workspace: str | Path,
    manifest_path: str | Path,
    member_limit: int = 1,
    nonmember_limit: int = 1,
    seed: int | None = None,
) -> dict[str, Any]:
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    manifest = _load_prepare_manifest(manifest_path)
    if manifest.get("status") != "ready":
        blocked_summary = {
            "status": "blocked",
            "track": "defense",
            "method": "privacy-aware-adapter",
            "mode": "run-pilot",
            "contract_stage": "run-pilot",
            "gpu_release": "none",
            "admitted_change": "none",
            "blocker_reason": manifest.get("blocker_reason", "prepare-manifest-not-ready"),
            "workspace": str(workspace_path),
            "manifest_path": str(manifest_path),
            "manifest": manifest,
        }
        summary_path = workspace_path / "summary.json"
        summary_path.write_text(
            json.dumps(blocked_summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return blocked_summary

    member_limit = max(1, int(member_limit))
    nonmember_limit = max(1, int(nonmember_limit))

    member_stage = _stage_dataset_subset(
        manifest["dataset_contract"]["member_dataset"]["path"],
        workspace_path / "packet" / "member",
        sample_limit=member_limit,
    )
    nonmember_stage = _stage_dataset_subset(
        manifest["dataset_contract"]["nonmember_dataset"]["path"],
        workspace_path / "packet" / "nonmember",
        sample_limit=nonmember_limit,
    )
    output_dir = workspace_path / "run-output"
    stdout_path = workspace_path / "train.stdout.txt"
    stderr_path = workspace_path / "train.stderr.txt"

    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "train_smp_lora.py"
    runtime = manifest["runtime"]
    command = [
        sys.executable,
        str(script_path),
        "--local_model",
        str(manifest["baseline_checkpoint"]["resolved_checkpoint_dir"]),
        "--member_dir",
        str(member_stage["staged_root"]),
        "--nonmember_dir",
        str(nonmember_stage["staged_root"]),
        "--output_dir",
        str(output_dir),
        "--rank",
        str(runtime["rank"]),
        "--alpha",
        str(runtime["alpha"]),
        "--lambda_coeff",
        str(runtime["lambda_coeff"]),
        "--delta",
        str(runtime["delta"]),
        "--lora_lr",
        str(runtime["lora_lr"]),
        "--proxy_lr",
        str(runtime["proxy_lr"]),
        "--optimizer",
        str(runtime["optimizer"]),
        "--sgd_momentum",
        str(runtime["sgd_momentum"]),
        "--proxy_hidden_dim",
        str(runtime["proxy_hidden_dim"]),
        "--proxy_steps",
        str(runtime["proxy_steps"]),
        "--num_epochs",
        str(runtime["num_epochs"]),
        "--batch_size",
        str(runtime["batch_size"]),
        "--num_workers",
        str(runtime["num_workers"]),
        "--method",
        str(runtime["method"]),
        "--device",
        str(runtime["device"]),
        "--save_every",
        "1000000",
    ]
    if seed is not None:
        command.extend(["--seed", str(int(seed))])

    returncode, stdout_text, stderr_text = _run_h2_training_command(command, cwd=repo_root)
    stdout_path.write_text(stdout_text, encoding="utf-8")
    stderr_path.write_text(stderr_text, encoding="utf-8")

    final_dir = output_dir / "final"
    config_path = output_dir / "config.json"
    checkpoint_meta_path = final_dir / "checkpoint_meta.json"
    lora_summary_path = final_dir / "lora_summary.json"
    training_log_path = final_dir / "training_log.json"
    lora_weights_path = final_dir / "lora_weights.pt"
    proxy_weights_path = final_dir / "proxy_weights.pt"
    review_checkpoint_dir = workspace_path / "review-checkpoint"

    training_log = _read_json_file(training_log_path)
    lora_summary = _read_json_file(lora_summary_path)
    checkpoint_meta = _read_json_file(checkpoint_meta_path)
    config_payload = _read_json_file(config_path)

    checks = {
        "script_entrypoint_exists": script_path.exists(),
        "manifest_ready": True,
        "training_returncode_zero": returncode == 0,
        "config_written": config_path.exists(),
        "checkpoint_meta_written": checkpoint_meta_path.exists(),
        "lora_summary_written": lora_summary_path.exists(),
        "training_log_written": training_log_path.exists(),
        "lora_weights_written": lora_weights_path.exists(),
        "proxy_weights_written": proxy_weights_path.exists(),
    }
    review_checkpoint_path = None
    if all(
        checks[key]
        for key in ("training_returncode_zero", "lora_weights_written")
    ):
        try:
            review_checkpoint_path = _materialize_h2_review_checkpoint(
                baseline_checkpoint_dir=manifest["baseline_checkpoint"]["resolved_checkpoint_dir"],
                lora_weights_path=lora_weights_path,
                output_dir=review_checkpoint_dir,
                rank=runtime["rank"],
                alpha=runtime["alpha"],
            )
        except Exception:
            review_checkpoint_path = None
    checks["review_checkpoint_written"] = review_checkpoint_path is not None and Path(review_checkpoint_path).exists()
    status = "ready" if all(checks.values()) else "blocked"

    metrics: dict[str, Any] = {}
    if isinstance(training_log, list) and training_log:
        first_step = training_log[0]
        metrics.update(
            {
                "step0_adaptation_loss": first_step.get("adaptation_loss"),
                "step0_mi_gain": first_step.get("mi_gain"),
                "step0_proxy_loss": first_step.get("proxy_loss"),
                "step0_objective": first_step.get("objective"),
            }
        )
    if isinstance(lora_summary, dict):
        metrics["total_lora_params"] = lora_summary.get("total_lora_params")
        metrics["num_lora_layers"] = lora_summary.get("num_lora_layers")

    summary = {
        "status": status,
        "track": "defense",
        "method": "privacy-aware-adapter",
        "paper": "SMP-LoRA_AAAI2025",
        "mode": "run-pilot",
        "contract_stage": "run-pilot",
        "gpu_release": "none",
        "admitted_change": "none",
        "provenance_status": manifest.get("provenance_status", "workspace-verified"),
        "workspace": str(workspace_path),
        "manifest_path": str(manifest_path),
        "baseline_checkpoint": manifest["baseline_checkpoint"],
        "defended_checkpoint": str(final_dir),
        "packet": manifest["packet"],
        "executed_packet": {
            "member_count": member_stage["count"],
            "nonmember_count": nonmember_stage["count"],
            "member_stage_dir": member_stage["staged_root"],
            "nonmember_stage_dir": nonmember_stage["staged_root"],
        },
        "runtime": {
            **runtime,
            "seed": int(seed) if seed is not None else None,
        },
        "checks": checks,
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
            "config": str(config_path),
            "checkpoint_meta": str(checkpoint_meta_path),
            "lora_summary": str(lora_summary_path),
            "training_log": str(training_log_path),
            "lora_weights": str(lora_weights_path),
            "proxy_weights": str(proxy_weights_path),
            "review_checkpoint": str(review_checkpoint_path) if review_checkpoint_path is not None else str(review_checkpoint_dir / "model.safetensors"),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        },
        "training_process": {
            "command": command,
            "cwd": str(repo_root),
            "returncode": int(returncode),
        },
        "config_payload": config_payload,
        "checkpoint_meta": checkpoint_meta,
        "metrics": metrics,
        "notes": [
            "This bounded run executes the first canonical H2 pilot but does not yet provide any attack-side review.",
            "A ready status here only means training artifacts were emitted under the prepared contract.",
        ],
    }
    if status != "ready":
        summary["blocker_reason"] = "training-artifacts-incomplete"

    summary_path = workspace_path / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary


def _load_run_pilot_summary(run_summary_path: str | Path) -> dict[str, Any]:
    path = Path(run_summary_path)
    if not path.exists():
        return {
            "status": "blocked",
            "blocker_reason": "run-summary-missing",
            "run_summary_path": str(path),
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "status": "blocked",
            "blocker_reason": "run-summary-invalid",
            "run_summary_path": str(path),
            "error": str(exc),
        }
    if payload.get("status") != "ready" or payload.get("mode") != "run-pilot":
        return {
            "status": "blocked",
            "blocker_reason": "run-summary-not-ready",
            "run_summary_path": str(path),
            "run_summary": payload,
        }
    return payload


def review_h2_defense_pilot(
    *,
    workspace: str | Path,
    run_summary_path: str | Path,
    shadow_reference_summary: str | Path = DEFAULT_H2_SHADOW_REFERENCE_SUMMARY,
    device: str = "cpu",
    noise_seed: int | None = None,
    provenance_status: str = "workspace-verified",
) -> dict[str, Any]:
    from diffaudit.attacks.gsa import _extract_gsa_loss_scores, evaluate_gsa_loss_score_packet
    from diffaudit.defenses.risk_targeted_unlearning import (
        _load_shadow_reference_summary,
        _write_review_packet_summary,
    )

    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    run_summary = _load_run_pilot_summary(run_summary_path)
    if run_summary.get("status") != "ready":
        blocked = {
            "status": "blocked",
            "track": "defense",
            "method": "privacy-aware-adapter",
            "mode": "review-pilot",
            "contract_stage": "review-pilot",
            "gpu_release": "none",
            "admitted_change": "none",
            "blocker_reason": run_summary.get("blocker_reason", "run-summary-not-ready"),
            "workspace": str(workspace_path),
            "run_summary": run_summary,
        }
        summary_path = workspace_path / "summary.json"
        summary_path.write_text(json.dumps(blocked, indent=2, ensure_ascii=True), encoding="utf-8")
        return blocked

    shadow_reference = _load_shadow_reference_summary(shadow_reference_summary)

    runtime_defaults = shadow_reference.get("runtime", {}) or {}
    resolution = int(runtime_defaults.get("resolution", 32))
    ddpm_num_steps = int(runtime_defaults.get("ddpm_num_steps", 20))
    sampling_frequency = int(runtime_defaults.get("sampling_frequency", 2))
    attack_method = int(runtime_defaults.get("attack_method", 1))
    prediction_type = str(runtime_defaults.get("prediction_type", "epsilon"))

    executed_packet = run_summary["executed_packet"]
    member_dir = executed_packet["member_stage_dir"]
    nonmember_dir = executed_packet["nonmember_stage_dir"]
    baseline_checkpoint_root = run_summary["baseline_checkpoint"]["checkpoint_root"]
    baseline_checkpoint_dir = run_summary["baseline_checkpoint"]["resolved_checkpoint_dir"]
    review_checkpoint_path = (
        (run_summary.get("artifact_paths", {}) or {}).get("review_checkpoint")
        if isinstance(run_summary.get("artifact_paths"), dict)
        else None
    )
    defended_checkpoint_dir = (
        str(Path(review_checkpoint_path).parent)
        if review_checkpoint_path is not None
        else run_summary["defended_checkpoint"]
    )

    export_root = workspace_path / "exports"
    export_root.mkdir(parents=True, exist_ok=True)

    baseline_member_export = _extract_gsa_loss_scores(
        dataset_dir=member_dir,
        checkpoint_root=baseline_checkpoint_root,
        checkpoint_dir=baseline_checkpoint_dir,
        output_path=export_root / "baseline-target-member-loss-scores.pt",
        records_path=export_root / "baseline-target-member-loss-scores.jsonl",
        resolution=resolution,
        ddpm_num_steps=ddpm_num_steps,
        sampling_frequency=sampling_frequency,
        attack_method=attack_method,
        prediction_type=prediction_type,
        device=device,
        noise_seed=noise_seed,
    )
    baseline_nonmember_export = _extract_gsa_loss_scores(
        dataset_dir=nonmember_dir,
        checkpoint_root=baseline_checkpoint_root,
        checkpoint_dir=baseline_checkpoint_dir,
        output_path=export_root / "baseline-target-nonmember-loss-scores.pt",
        records_path=export_root / "baseline-target-nonmember-loss-scores.jsonl",
        resolution=resolution,
        ddpm_num_steps=ddpm_num_steps,
        sampling_frequency=sampling_frequency,
        attack_method=attack_method,
        prediction_type=prediction_type,
        device=device,
        noise_seed=noise_seed,
    )
    defended_member_export = _extract_gsa_loss_scores(
        dataset_dir=member_dir,
        checkpoint_root=Path(defended_checkpoint_dir).parent,
        checkpoint_dir=defended_checkpoint_dir,
        output_path=export_root / "defended-target-member-loss-scores.pt",
        records_path=export_root / "defended-target-member-loss-scores.jsonl",
        resolution=resolution,
        ddpm_num_steps=ddpm_num_steps,
        sampling_frequency=sampling_frequency,
        attack_method=attack_method,
        prediction_type=prediction_type,
        device=device,
        noise_seed=noise_seed,
    )
    defended_nonmember_export = _extract_gsa_loss_scores(
        dataset_dir=nonmember_dir,
        checkpoint_root=Path(defended_checkpoint_dir).parent,
        checkpoint_dir=defended_checkpoint_dir,
        output_path=export_root / "defended-target-nonmember-loss-scores.pt",
        records_path=export_root / "defended-target-nonmember-loss-scores.jsonl",
        resolution=resolution,
        ddpm_num_steps=ddpm_num_steps,
        sampling_frequency=sampling_frequency,
        attack_method=attack_method,
        prediction_type=prediction_type,
        device=device,
        noise_seed=noise_seed,
    )

    baseline_packet_summary_path = workspace_path / "baseline-review-packet-summary.json"
    defended_packet_summary_path = workspace_path / "defended-review-packet-summary.json"
    _write_review_packet_summary(
        output_path=baseline_packet_summary_path,
        shadow_reference_summary=shadow_reference,
        target_member_export=baseline_member_export,
        target_nonmember_export=baseline_nonmember_export,
        label="baseline-target-subset",
        sample_id_allowlist=[],
        checkpoint_dir=baseline_checkpoint_dir,
    )
    _write_review_packet_summary(
        output_path=defended_packet_summary_path,
        shadow_reference_summary=shadow_reference,
        target_member_export=defended_member_export,
        target_nonmember_export=defended_nonmember_export,
        label="defended-target-subset",
        sample_id_allowlist=[],
        checkpoint_dir=defended_checkpoint_dir,
    )

    baseline_eval = evaluate_gsa_loss_score_packet(
        workspace=workspace_path / "baseline-threshold-eval",
        packet_summary=baseline_packet_summary_path,
        provenance_status=provenance_status,
    )
    defended_eval = evaluate_gsa_loss_score_packet(
        workspace=workspace_path / "defended-threshold-eval",
        packet_summary=defended_packet_summary_path,
        provenance_status=provenance_status,
    )

    metric_names = ("auc", "asr", "tpr_at_1pct_fpr", "tpr_at_0_1pct_fpr")
    baseline_metrics = baseline_eval["target_transfer"]["metrics"]
    defended_metrics = defended_eval["target_transfer"]["metrics"]
    metric_deltas = {
        metric: _round6(float(defended_metrics[metric]) - float(baseline_metrics[metric]))
        for metric in metric_names
    }

    summary = {
        "status": "ready",
        "track": "defense",
        "method": "privacy-aware-adapter",
        "paper": "SMP-LoRA_AAAI2025",
        "mode": "review-pilot",
        "contract_stage": "review-pilot",
        "gpu_release": "none",
        "admitted_change": "none",
        "provenance_status": provenance_status,
        "workspace": str(workspace_path),
        "workspace_name": workspace_path.name,
        "run_summary_path": str(run_summary_path),
        "shadow_reference_summary": str(shadow_reference_summary),
        "review_surface": "same-packet target-subset threshold-transfer",
        "attacker_mode": "transfer-only-shadow-threshold",
        "low_fpr_read_order": ["tpr_at_0_1pct_fpr", "tpr_at_1pct_fpr", "auc", "asr"],
        "baseline": {
            "checkpoint_dir": str(baseline_checkpoint_dir),
            "member_export": baseline_member_export,
            "nonmember_export": baseline_nonmember_export,
            "threshold_eval": baseline_eval,
        },
        "defended": {
            "checkpoint_dir": str(defended_checkpoint_dir),
            "member_export": defended_member_export,
            "nonmember_export": defended_nonmember_export,
            "threshold_eval": defended_eval,
        },
        "comparison": {
            "baseline_metrics": baseline_metrics,
            "defended_metrics": defended_metrics,
            "metric_deltas": metric_deltas,
        },
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
            "baseline_packet_summary": str(baseline_packet_summary_path),
            "defended_packet_summary": str(defended_packet_summary_path),
            "baseline_threshold_eval_summary": str(workspace_path / "baseline-threshold-eval" / "summary.json"),
            "defended_threshold_eval_summary": str(workspace_path / "defended-threshold-eval" / "summary.json"),
        },
        "notes": [
            "This review reuses an existing undefended shadow export only for threshold-transfer diagnostics.",
            "It is same-packet and attack-side readable, but still transfer-only rather than defense-aware.",
        ],
    }
    summary_path = workspace_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary
