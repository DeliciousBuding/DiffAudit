"""Adapter layer for the vendored SecMI implementation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
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
