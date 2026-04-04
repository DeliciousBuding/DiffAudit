"""Planning helpers for integrating the official SecMI attack flow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from diffaudit.config import AuditConfig


@dataclass(frozen=True)
class SecmiPlan:
    entrypoint: str
    dataset: str
    data_root: str
    model_dir: str
    t_sec: int
    k: int
    batch_size: int


@dataclass(frozen=True)
class SecmiArtifacts:
    checkpoint_path: str
    flagfile_path: str


@dataclass(frozen=True)
class SecmiRunnerSpec:
    repo_root: str
    entrypoint_path: str
    python_module: str
    checkpoint_path: str
    flagfile_path: str
    dataset: str
    data_root: str
    t_sec: int
    k: int
    batch_size: int


REQUIRED_SECMI_WORKSPACE_FILES = (
    "mia_evals/secmia.py",
    "mia_evals/dataset_utils.py",
    "model.py",
    "diffusion.py",
)

SECMI_MEMBER_SPLIT_FILENAMES = {
    "CIFAR10": "CIFAR10_train_ratio0.5.npz",
    "CIFAR100": "CIFAR100_train_ratio0.5.npz",
    "STL10-U": "STL10_Unlabeled_train_ratio0.5.npz",
    "TINY-IN": "TINY-IN_train_ratio0.5.npz",
}


def build_secmi_plan(config: AuditConfig) -> SecmiPlan:
    if config.attack.method != "secmi":
        raise ValueError(f"Unsupported attack method for SecMI plan: {config.attack.method}")

    dataset = config.assets.dataset_name
    data_root = config.assets.dataset_root
    model_dir = config.assets.model_dir
    params = config.attack.parameters

    if not dataset:
        raise ValueError("SecMI requires assets.dataset_name")
    if not data_root:
        raise ValueError("SecMI requires assets.dataset_root")
    if not model_dir:
        raise ValueError("SecMI requires assets.model_dir")
    if "t_sec" not in params:
        raise ValueError("SecMI requires attack.parameters.t_sec")
    if "k" not in params:
        raise ValueError("SecMI requires attack.parameters.k")

    return SecmiPlan(
        entrypoint="mia_evals/secmia.py",
        dataset=dataset,
        data_root=data_root,
        model_dir=model_dir,
        t_sec=int(params["t_sec"]),
        k=int(params["k"]),
        batch_size=int(params.get("batch_size", 32)),
    )


def resolve_secmi_artifacts(model_dir: str | Path) -> SecmiArtifacts:
    model_path = Path(model_dir)
    flagfile_path = model_path / "flagfile.txt"
    if not flagfile_path.exists():
        raise FileNotFoundError(f"Missing SecMI flagfile: {flagfile_path}")

    checkpoint_path = model_path / "checkpoint.pt"
    if not checkpoint_path.exists():
        candidates = sorted(model_path.glob("ckpt-step*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No SecMI checkpoint found in {model_path}")
        checkpoint_path = candidates[-1]

    return SecmiArtifacts(
        checkpoint_path=str(checkpoint_path),
        flagfile_path=str(flagfile_path),
    )


def validate_secmi_workspace(workspace_dir: str | Path) -> dict[str, str]:
    workspace_path = Path(workspace_dir)
    missing = [
        relative_path
        for relative_path in REQUIRED_SECMI_WORKSPACE_FILES
        if not (workspace_path / relative_path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"SecMI workspace is missing required files: {', '.join(missing)}"
        )

    return {
        "status": "ready",
        "workspace_dir": str(workspace_path),
        "entrypoint": str(workspace_path / "mia_evals" / "secmia.py"),
    }


def resolve_secmi_member_split_path(
    dataset: str,
    member_split_root: str | Path,
) -> Path | None:
    split_name = SECMI_MEMBER_SPLIT_FILENAMES.get(dataset.upper())
    if split_name is None:
        return None
    return Path(member_split_root) / split_name


def probe_secmi_assets(
    dataset: str,
    dataset_root: str | Path,
    model_dir: str | Path,
    member_split_root: str | Path,
) -> dict[str, object]:
    dataset_root_path = Path(dataset_root)
    model_dir_path = Path(model_dir)
    member_split_root_path = Path(member_split_root)
    flagfile_path = model_dir_path / "flagfile.txt"
    checkpoint_path = model_dir_path / "checkpoint.pt"
    ckpt_step_candidates = sorted(model_dir_path.glob("ckpt-step*.pt"))
    resolved_checkpoint_path = checkpoint_path if checkpoint_path.exists() else (
        ckpt_step_candidates[-1] if ckpt_step_candidates else checkpoint_path
    )
    member_split_path = resolve_secmi_member_split_path(dataset, member_split_root_path)

    checkpoint_exists = resolved_checkpoint_path.exists()
    flagfile_exists = flagfile_path.exists()
    dataset_root_exists = dataset_root_path.exists()
    member_split_exists = bool(member_split_path and member_split_path.exists())

    checks = {
        "checkpoint": checkpoint_exists,
        "flagfile": flagfile_exists,
        "dataset_root": dataset_root_exists,
        "member_split": member_split_exists,
    }
    status = "ready" if all(checks.values()) else "blocked"
    paths = {
        "model_dir": str(model_dir_path),
        "dataset_root": str(dataset_root_path),
        "member_split_root": str(member_split_root_path),
        "flagfile": str(flagfile_path),
        "checkpoint": str(resolved_checkpoint_path),
        "member_split": str(member_split_path) if member_split_path else "",
    }
    labels = {
        "checkpoint": "checkpoint",
        "flagfile": "flagfile",
        "dataset_root": "dataset_root",
        "member_split": "member split",
    }
    missing_keys = [name for name, is_ready in checks.items() if not is_ready]
    missing = [paths[name] or labels[name] for name in missing_keys]

    return {
        "status": status,
        "dataset": dataset,
        "checks": checks,
        "paths": paths,
        "missing": missing,
        "missing_keys": missing_keys,
        "missing_items": [Path(item).name for item in missing],
        "missing_description": " / ".join(labels[name] for name in missing_keys),
    }


def build_secmi_runner_spec(
    plan: SecmiPlan,
    artifacts: SecmiArtifacts,
    repo_root: str | Path,
) -> SecmiRunnerSpec:
    repo_path = Path(repo_root)
    entrypoint_path = repo_path / plan.entrypoint
    if not entrypoint_path.exists():
        raise FileNotFoundError(f"SecMI entrypoint not found: {entrypoint_path}")

    return SecmiRunnerSpec(
        repo_root=str(repo_path),
        entrypoint_path=str(entrypoint_path),
        python_module="mia_evals.secmia",
        checkpoint_path=artifacts.checkpoint_path,
        flagfile_path=artifacts.flagfile_path,
        dataset=plan.dataset,
        data_root=plan.data_root,
        t_sec=plan.t_sec,
        k=plan.k,
        batch_size=plan.batch_size,
    )


def explain_secmi_assets(
    config: AuditConfig,
    member_split_root: str | Path = "third_party/secmi/mia_evals/member_splits",
) -> dict[str, object]:
    plan = build_secmi_plan(config)
    summary = probe_secmi_assets(
        dataset=plan.dataset,
        dataset_root=plan.data_root,
        model_dir=plan.model_dir,
        member_split_root=member_split_root,
    )
    return {
        **summary,
        "model_dir": plan.model_dir,
        "data_root": plan.data_root,
    }
