"""Adapter layer for the vendored SecMI implementation."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

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
    from third_party.secmi.model import UNet

    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    flagfile_path = target_path / "flagfile.txt"
    shutil.copyfile(flagfile_source, flagfile_path)

    model = UNet(T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1)
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


def probe_secmi_runtime(config: AuditConfig, repo_root: str) -> tuple[int, dict[str, Any]]:
    try:
        context = prepare_secmi_adapter(config, repo_root)
        flags_obj = context.module.get_FLAGS(context.artifacts.flagfile_path)
        model = context.module.get_model(context.artifacts.checkpoint_path, flags_obj, WA=True)
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
            "checkpoint_loaded",
            "model_instantiated",
        ],
        "model_type": type(model).__name__,
        **summarize_secmi_adapter(context),
    }
