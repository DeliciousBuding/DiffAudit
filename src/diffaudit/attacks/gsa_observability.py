"""Read-only helpers for the Finding NeMo migrated DDPM observability contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from diffusers import UNet2DModel

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
