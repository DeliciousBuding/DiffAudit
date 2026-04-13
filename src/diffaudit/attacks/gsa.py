"""Runtime helpers for the white-box GSA DDPM workflow."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


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
    return any(path.is_file() for path in dataset_path.rglob("*"))


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


def _preprocess(member: torch.Tensor, nonmember: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
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
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    shadow_member = torch.cat([_load_gradient_tensor(path) for path in shadow_member_paths], dim=0)
    shadow_nonmember = torch.cat([_load_gradient_tensor(path) for path in shadow_nonmember_paths], dim=0)
    target_member = _load_gradient_tensor(target_member_path)
    target_nonmember = _load_gradient_tensor(target_nonmember_path)

    shadow_x, shadow_y = _preprocess(shadow_member, shadow_nonmember)
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

    target_x, target_y = _preprocess(target_member, target_nonmember)
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
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported GSA runtime device: {device}")
    resolved_device = "cuda" if requested_device == "cuda" else "cpu"
    if requested_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

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
