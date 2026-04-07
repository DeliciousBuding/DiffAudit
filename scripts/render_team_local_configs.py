"""Render per-track local run configs from team-local asset bindings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping at {path}")
    return payload


def _required(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Missing required key: {'.'.join(keys)}")
        current = current[key]
    return current


def build_secmi_config(team_local: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": {"name": "secmi-plan", "model_family": "diffusion", "access_level": "black_box"},
        "assets": {
            "dataset_id": "local-secmi-dataset",
            "dataset_name": "cifar10",
            "dataset_root": _required(team_local, "gray_box", "secmi", "dataset_root"),
            "model_id": "local-secmi-model",
            "model_dir": _required(team_local, "gray_box", "secmi", "model_dir"),
        },
        "attack": {"method": "secmi", "num_samples": 8, "parameters": {"t_sec": 100, "k": 10, "batch_size": 16}},
        "report": {"output_dir": "experiments/secmi-plan-local"},
    }


def build_pia_config(team_local: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": {"name": "pia-plan", "model_family": "diffusion", "access_level": "semi_white_box"},
        "assets": {
            "dataset_id": "local-pia-dataset",
            "dataset_name": "cifar10",
            "dataset_root": _required(team_local, "gray_box", "pia", "dataset_parent"),
            "model_id": "local-pia-model",
            "model_dir": _required(team_local, "gray_box", "pia", "model_dir"),
        },
        "attack": {
            "method": "pia",
            "num_samples": 8,
            "parameters": {"attacker_name": "PIA", "attack_num": 30, "interval": 10, "batch_size": 64},
        },
        "report": {"output_dir": "experiments/pia-plan-local"},
    }


def build_variation_config(team_local: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": {"name": "variation-plan", "model_family": "diffusion", "access_level": "black_box"},
        "assets": {
            "dataset_id": "local-variation-query-set",
            "dataset_root": _required(team_local, "black_box", "variation", "query_image_root"),
            "model_id": "local-variation-api",
        },
        "attack": {
            "method": "variation",
            "num_samples": 8,
            "parameters": {
                "endpoint": _required(team_local, "black_box", "variation", "endpoint"),
                "num_averages": 3,
                "distance_metric": "l2",
                "threshold_strategy": "threshold",
            },
        },
        "report": {"output_dir": "experiments/variation-plan-local"},
    }


def build_recon_config(team_local: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": {"name": "recon-plan", "model_family": "diffusion", "access_level": "black_box"},
        "assets": {
            "dataset_id": "local-recon-query-dataset",
            "dataset_root": _required(team_local, "black_box", "recon", "query_dataset_pt"),
            "model_id": "local-recon-model",
            "model_dir": _required(team_local, "black_box", "recon", "lora_checkpoint_dir"),
        },
        "attack": {
            "method": "recon",
            "num_samples": 8,
            "parameters": {
                "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
                "num_validation_images": 3,
                "inference_steps": 30,
                "eval_method": "threshold",
            },
        },
        "report": {"output_dir": "experiments/recon-plan-local"},
    }


def build_clid_config(team_local: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": {"name": "clid-plan", "model_family": "diffusion", "access_level": "black_box"},
        "assets": {
            "dataset_id": "local-clid-dataset",
            "dataset_train_ref": _required(team_local, "black_box", "clid", "dataset_train_ref"),
            "dataset_test_ref": _required(team_local, "black_box", "clid", "dataset_test_ref"),
            "model_id": "local-clid-model",
            "model_dir": _required(team_local, "black_box", "clid", "model_dir"),
        },
        "attack": {
            "method": "clid",
            "num_samples": 8,
            "parameters": {
                "variant": "clid_impt",
                "max_n_samples": 3,
                "resolution": 512,
                "image_column": "image",
                "caption_column": "text",
            },
        },
        "report": {"output_dir": "experiments/clid-plan-local"},
    }


def render_all(team_local_path: Path, output_dir: Path) -> list[Path]:
    team_local = _load_yaml(team_local_path)
    rendered = {
        "secmi.local.yaml": build_secmi_config(team_local),
        "pia.local.yaml": build_pia_config(team_local),
        "variation.local.yaml": build_variation_config(team_local),
        "recon.local.yaml": build_recon_config(team_local),
        "clid.local.yaml": build_clid_config(team_local),
    }
    paths: list[Path] = []
    for filename, payload in rendered.items():
        path = output_dir / filename
        _write_yaml(path, payload)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-local", default="configs/assets/team.local.yaml")
    parser.add_argument("--output-dir", default="tmp/configs/rendered")
    args = parser.parse_args()

    for path in render_all(Path(args.team_local), Path(args.output_dir)):
        print(path.as_posix())


if __name__ == "__main__":
    main()
