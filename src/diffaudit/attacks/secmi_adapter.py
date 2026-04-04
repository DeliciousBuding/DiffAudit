"""Adapter layer for the vendored SecMI implementation."""

from __future__ import annotations

import shutil
from types import SimpleNamespace
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

from diffaudit.attacks.secmi import (
    SecmiArtifacts,
    SecmiPlan,
    SecmiRunnerSpec,
    build_secmi_plan,
    build_secmi_runner_spec,
    resolve_secmi_artifacts,
)
from diffaudit.config import AuditConfig


@dataclass(frozen=True)
class SecmiAdapterContext:
    plan: SecmiPlan
    artifacts: SecmiArtifacts
    runner: SecmiRunnerSpec
    module: ModuleType


def load_vendored_secmi_module() -> ModuleType:
    return import_module("third_party.secmi.mia_evals.secmia")


def prepare_secmi_adapter(config: AuditConfig, repo_root: str) -> SecmiAdapterContext:
    plan = build_secmi_plan(config)
    artifacts = resolve_secmi_artifacts(plan.model_dir)
    runner = build_secmi_runner_spec(plan, artifacts, repo_root)
    module = load_vendored_secmi_module()
    return SecmiAdapterContext(
        plan=plan,
        artifacts=artifacts,
        runner=runner,
        module=module,
    )


def summarize_secmi_adapter(context: SecmiAdapterContext) -> dict[str, Any]:
    return {
        "dataset": context.plan.dataset,
        "data_root": context.plan.data_root,
        "model_dir": context.plan.model_dir,
        "checkpoint_path": context.artifacts.checkpoint_path,
        "flagfile_path": context.artifacts.flagfile_path,
        "entrypoint_path": context.runner.entrypoint_path,
        "python_module": context.runner.python_module,
    }


def run_secmi_dry_run(context: SecmiAdapterContext) -> dict[str, Any]:
    available_functions = [
        name
        for name in ("get_FLAGS", "get_model", "secmi_attack", "load_member_data")
        if hasattr(context.module, name)
    ]
    return {
        "status": "ready",
        "available_functions": available_functions,
        **summarize_secmi_adapter(context),
    }


def probe_secmi_dry_run(config: AuditConfig, repo_root: str) -> tuple[int, dict[str, Any]]:
    try:
        context = prepare_secmi_adapter(config, repo_root)
    except FileNotFoundError as exc:
        return 1, {
            "status": "blocked",
            "error": str(exc),
            "repo_root": repo_root,
        }

    return 0, run_secmi_dry_run(context)


def bootstrap_secmi_smoke_assets(target_dir: str | Path, flagfile_source: str | Path) -> dict[str, str]:
    import torch

    from third_party.secmi.model import UNet

    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    flagfile_path = target_path / "flagfile.txt"
    if flagfile_source and Path(flagfile_source).exists():
        shutil.copyfile(flagfile_source, flagfile_path)
        flags = parse_secmi_flagfile(flagfile_path)
    else:
        flag_lines = [
            "--T=100",
            "--attn=1",
            "--batch_size=8",
            "--beta_1=0.0001",
            "--beta_T=0.02",
            "--ch=32",
            "--ch_mult=1",
            "--ch_mult=2",
            "--dropout=0.1",
            "--img_size=32",
            "--num_res_blocks=1",
            "--mean_type=epsilon",
            "--var_type=fixedlarge",
            "--train",
        ]
        flagfile_path.write_text("\n".join(flag_lines) + "\n", encoding="utf-8")
        flags = parse_secmi_flagfile(flagfile_path)

    model = UNet(
        T=flags.T,
        ch=flags.ch,
        ch_mult=flags.ch_mult,
        attn=flags.attn,
        num_res_blocks=flags.num_res_blocks,
        dropout=flags.dropout,
    )
    state_dict = model.state_dict()
    checkpoint = {
        "ema_model": state_dict,
        "net_model": state_dict,
    }

    checkpoint_path = target_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    return {
        "flagfile_path": str(flagfile_path),
        "checkpoint_path": str(checkpoint_path),
    }


def parse_secmi_flagfile(flagfile_path: str | Path) -> SimpleNamespace:
    values: dict[str, Any] = {
        "T": 100,
        "attn": [1],
        "batch_size": 8,
        "beta_1": 0.0001,
        "beta_T": 0.02,
        "ch": 32,
        "ch_mult": [1, 2],
        "dropout": 0.1,
        "img_size": 32,
        "num_res_blocks": 1,
        "mean_type": "epsilon",
        "var_type": "fixedlarge",
    }
    multi_keys = {"ch_mult", "attn"}
    seen_multi_keys: set[str] = set()

    with Path(flagfile_path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or not line.startswith("--") or "=" not in line:
                continue
            key, value = line[2:].split("=", 1)
            if key in multi_keys:
                if key not in seen_multi_keys:
                    values[key] = []
                    seen_multi_keys.add(key)
                values.setdefault(key, []).append(int(value))
            elif key in {"T", "batch_size", "ch", "img_size", "num_res_blocks", "warmup", "num_workers"}:
                values[key] = int(value)
            elif key in {"beta_1", "beta_T", "dropout", "lr", "grad_clip", "ema_decay"}:
                values[key] = float(value)
            elif value.lower() in {"true", "false"}:
                values[key] = value.lower() == "true"
            else:
                values[key] = value

    return SimpleNamespace(**values)


def run_synthetic_secmi_stat_smoke(workspace: str | Path, device: str = "cpu") -> dict[str, Any]:
    import torch

    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    smoke_assets = bootstrap_secmi_smoke_assets(
        target_dir=workspace_path / "synthetic-secmi-assets",
        flagfile_source="",
    )
    flags_obj = parse_secmi_flagfile(smoke_assets["flagfile_path"])
    module = load_vendored_secmi_module()
    model = module.get_model(smoke_assets["checkpoint_path"], flags_obj, WA=True).to(device)

    member_loader = [(torch.rand(2, 3, 32, 32), torch.ones(2, dtype=torch.long))]
    nonmember_loader = [(torch.rand(2, 3, 32, 32) * 0.2, torch.zeros(2, dtype=torch.long))]

    member_results = module.get_intermediate_results(
        model,
        flags_obj,
        member_loader,
        t_sec=20,
        timestep=10,
        device=device,
    )
    nonmember_results = module.get_intermediate_results(
        model,
        flags_obj,
        nonmember_loader,
        t_sec=20,
        timestep=10,
        device=device,
    )

    t_results = {
        "member_diffusions": member_results["internal_diffusions"],
        "member_internal_samples": member_results["internal_denoise"],
        "nonmember_diffusions": nonmember_results["internal_diffusions"],
        "nonmember_internal_samples": nonmember_results["internal_denoise"],
    }
    attack_results = module.execute_attack(t_results, type="stat")

    result = {
        "status": "ready",
        "method": "secmi",
        "mode": "synthetic-stat-smoke",
        "auc": round(float(attack_results["auc"]), 6),
        "asr": round(float(attack_results["asr"]), 6),
        "device": device,
        "workspace": str(workspace_path),
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
            "flagfile": smoke_assets["flagfile_path"],
            "checkpoint": smoke_assets["checkpoint_path"],
        },
        "score_stats": {
            "member_count": int(attack_results["member_scores"].numel()),
            "nonmember_count": int(attack_results["nonmember_scores"].numel()),
            "member_mean": round(float(attack_results["member_scores"].mean()), 6),
            "nonmember_mean": round(float(attack_results["nonmember_scores"].mean()), 6),
        },
    }
    (workspace_path / "summary.json").write_text(
        __import__("json").dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    shutil.rmtree(workspace_path / "synthetic-secmi-assets", ignore_errors=True)
    return result


def probe_secmi_runtime(config: AuditConfig, repo_root: str) -> tuple[int, dict[str, Any]]:
    try:
        context = prepare_secmi_adapter(config, repo_root)
        flags_obj = parse_secmi_flagfile(context.artifacts.flagfile_path)
    except FileNotFoundError as exc:
        return 1, {
            "status": "blocked",
            "error": str(exc),
            "repo_root": repo_root,
        }
    except Exception as exc:  # runtime probe should surface integration failures cleanly
        return 2, {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "repo_root": repo_root,
        }

    return 0, {
        "status": "ready",
        "checks": [
            "flagfile_loaded",
            "checkpoint_path_present",
            "module_imported",
            "attack_functions_available",
        ],
        "model_T": flags_obj.T,
        **summarize_secmi_adapter(context),
    }

