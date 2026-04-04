"""Planning helpers for integrating the official SecMI attack flow."""

from __future__ import annotations

from dataclasses import dataclass

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
