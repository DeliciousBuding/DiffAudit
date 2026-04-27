"""CPU-first prep surface for 04-H1 risk-targeted unlearning pilots."""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np

from diffaudit.attacks.crossbox_pairboard import _align_surfaces, load_pairboard_surface
from diffaudit.attacks.gsa import _extract_gsa_loss_scores, evaluate_gsa_loss_score_packet


def _round6(value: float) -> float:
    return round(float(value), 6)


def _orient_memberness(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    if float(member_scores.mean()) < float(nonmember_scores.mean()):
        return -member_scores, -nonmember_scores, "negated"
    return member_scores.copy(), nonmember_scores.copy(), "identity"


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("Percentile ranking expects a 1D score vector")
    count = int(values.shape[0])
    if count == 0:
        raise ValueError("Percentile ranking requires at least one value")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.zeros(count, dtype=float)

    start = 0
    while start < count:
        end = start + 1
        while end < count and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average_rank / float(count)
        start = end
    return ranks


def _resolve_top_fraction_mask(ranks: np.ndarray, top_fraction: float) -> np.ndarray:
    if not 0.0 < float(top_fraction) <= 1.0:
        raise ValueError(f"top_fraction must lie in (0, 1], got {top_fraction}")
    count = int(ranks.shape[0])
    top_count = max(1, int(math.ceil(count * float(top_fraction))))
    order = np.argsort(-ranks, kind="mergesort")
    mask = np.zeros(count, dtype=bool)
    mask[order[:top_count]] = True
    return mask


def _prepare_records(
    indices: list[int],
    surface_a_scores: np.ndarray,
    surface_b_scores: np.ndarray,
    surface_a_ranks: np.ndarray,
    surface_b_ranks: np.ndarray,
    combined_risk: np.ndarray,
    top_fraction_intersection: np.ndarray,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for position, split_index in enumerate(indices):
        records.append(
            {
                "split_index": int(split_index),
                "position": int(position),
                "surface_a_score": _round6(surface_a_scores[position]),
                "surface_b_score": _round6(surface_b_scores[position]),
                "surface_a_percentile": _round6(surface_a_ranks[position]),
                "surface_b_percentile": _round6(surface_b_ranks[position]),
                "combined_risk": _round6(combined_risk[position]),
                "in_top_fraction_intersection": bool(top_fraction_intersection[position]),
            }
        )
    records.sort(
        key=lambda row: (
            -float(row["combined_risk"]),
            -float(row["surface_a_percentile"]),
            -float(row["surface_b_percentile"]),
            int(row["split_index"]),
        )
    )
    for rank, row in enumerate(records, start=1):
        row["combined_rank"] = int(rank)
    return records


def _match_nonmembers(
    selected_member_records: list[dict[str, Any]],
    nonmember_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    available = [dict(row) for row in nonmember_records]
    matches: list[dict[str, Any]] = []
    for member_row in selected_member_records:
        target = float(member_row["combined_risk"])
        best_index = min(
            range(len(available)),
            key=lambda idx: (
                abs(float(available[idx]["combined_risk"]) - target),
                abs(
                    float(available[idx].get("surface_a_percentile", available[idx]["combined_risk"]))
                    - float(member_row.get("surface_a_percentile", member_row["combined_risk"]))
                ),
                abs(
                    float(available[idx].get("surface_b_percentile", available[idx]["combined_risk"]))
                    - float(member_row.get("surface_b_percentile", member_row["combined_risk"]))
                ),
                int(available[idx]["split_index"]),
            ),
        )
        matches.append(available.pop(best_index))
    return matches


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )


def _write_index_file(path: Path, indices: list[int]) -> None:
    path.write_text("\n".join(str(int(index)) for index in indices) + "\n", encoding="utf-8")


def _read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_risk_targeted_unlearning_prep(
    *,
    workspace: str | Path,
    surface_a_path: str | Path,
    surface_b_path: str | Path,
    surface_a_name: str | None = None,
    surface_b_name: str | None = None,
    surface_a_family: str | None = None,
    surface_b_family: str | None = None,
    weight_a: float = 0.5,
    weight_b: float = 0.5,
    top_fraction: float = 0.1,
    top_k_values: list[int] | tuple[int, ...] = (16, 32, 64),
    provenance_status: str = "workspace-verified",
) -> dict[str, Any]:
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    if float(weight_a) < 0.0 or float(weight_b) < 0.0:
        raise ValueError("Risk aggregation weights must be non-negative")
    weight_total = float(weight_a) + float(weight_b)
    if weight_total <= 0.0:
        raise ValueError("At least one risk aggregation weight must be positive")
    norm_weight_a = float(weight_a) / weight_total
    norm_weight_b = float(weight_b) / weight_total

    unique_top_k = sorted({int(value) for value in top_k_values})
    if not unique_top_k or unique_top_k[0] <= 0:
        raise ValueError("top_k_values must contain at least one positive integer")

    surface_a = load_pairboard_surface(
        surface_a_path,
        name=surface_a_name,
        family=surface_a_family,
    )
    surface_b = load_pairboard_surface(
        surface_b_path,
        name=surface_b_name,
        family=surface_b_family,
    )
    aligned = _align_surfaces(surface_a, surface_b)

    member_indices = aligned["member_indices"]
    nonmember_indices = aligned["nonmember_indices"]
    if member_indices is None or nonmember_indices is None:
        raise ValueError("Risk-targeted prep requires explicit shared member and nonmember indices")
    if unique_top_k[-1] > len(member_indices) or unique_top_k[-1] > len(nonmember_indices):
        raise ValueError("Requested top-k exceeds aligned member or nonmember capacity")

    member_a, nonmember_a, orientation_a = _orient_memberness(
        aligned["member_a"], aligned["nonmember_a"]
    )
    member_b, nonmember_b, orientation_b = _orient_memberness(
        aligned["member_b"], aligned["nonmember_b"]
    )

    member_rank_a = _percentile_ranks(member_a)
    member_rank_b = _percentile_ranks(member_b)
    nonmember_rank_a = _percentile_ranks(nonmember_a)
    nonmember_rank_b = _percentile_ranks(nonmember_b)

    member_combined = (norm_weight_a * member_rank_a) + (norm_weight_b * member_rank_b)
    nonmember_combined = (norm_weight_a * nonmember_rank_a) + (norm_weight_b * nonmember_rank_b)

    member_top_intersection = _resolve_top_fraction_mask(member_rank_a, top_fraction) & _resolve_top_fraction_mask(
        member_rank_b, top_fraction
    )
    nonmember_top_intersection = _resolve_top_fraction_mask(
        nonmember_rank_a, top_fraction
    ) & _resolve_top_fraction_mask(nonmember_rank_b, top_fraction)

    member_records = _prepare_records(
        member_indices,
        member_a,
        member_b,
        member_rank_a,
        member_rank_b,
        member_combined,
        member_top_intersection,
    )
    nonmember_records = _prepare_records(
        nonmember_indices,
        nonmember_a,
        nonmember_b,
        nonmember_rank_a,
        nonmember_rank_b,
        nonmember_combined,
        nonmember_top_intersection,
    )

    member_records_path = workspace_path / "member-risk-records.jsonl"
    nonmember_records_path = workspace_path / "nonmember-risk-records.jsonl"
    _write_jsonl(member_records_path, member_records)
    _write_jsonl(nonmember_records_path, nonmember_records)

    forget_member_index_files: dict[str, str] = {}
    matched_nonmember_index_files: dict[str, str] = {}
    forget_ladders: dict[str, dict[str, Any]] = {}

    intersection_member_records = [row for row in member_records if row["in_top_fraction_intersection"]]

    for k in unique_top_k:
        ladder_key = f"k{k}"
        if len(intersection_member_records) >= k:
            selection_mode = "top-fraction-intersection"
            selected_member_records = intersection_member_records[:k]
        else:
            selection_mode = "aggregate-percentile"
            selected_member_records = member_records[:k]

        matched_nonmember_records = _match_nonmembers(selected_member_records, nonmember_records)
        selected_member_indices = [int(row["split_index"]) for row in selected_member_records]
        matched_nonmember_indices = [int(row["split_index"]) for row in matched_nonmember_records]

        member_index_file = workspace_path / f"forget-members-{ladder_key}.txt"
        nonmember_index_file = workspace_path / f"matched-nonmembers-{ladder_key}.txt"
        _write_index_file(member_index_file, selected_member_indices)
        _write_index_file(nonmember_index_file, matched_nonmember_indices)

        forget_member_index_files[ladder_key] = str(member_index_file)
        matched_nonmember_index_files[ladder_key] = str(nonmember_index_file)
        forget_ladders[ladder_key] = {
            "k": int(k),
            "selection_mode": selection_mode,
            "member_indices": selected_member_indices,
            "matched_nonmember_indices": matched_nonmember_indices,
            "retained_member_count": int(len(member_indices) - k),
            "retained_nonmember_count": int(len(nonmember_indices) - k),
            "mean_forget_member_risk": _round6(
                np.mean([float(row["combined_risk"]) for row in selected_member_records])
            ),
            "mean_matched_nonmember_risk": _round6(
                np.mean([float(row["combined_risk"]) for row in matched_nonmember_records])
            ),
        }

    summary = {
        "status": "ready",
        "track": "defense",
        "method": "risk-targeted-siss-prep",
        "paper": "Diffusion_Unlearning_SISS_ICLR2025",
        "mode": "risk-targeted-unlearning-prep",
        "device": "cpu",
        "gpu_release": "none",
        "admitted_change": "none",
        "provenance_status": provenance_status,
        "workspace": str(workspace_path),
        "workspace_name": workspace_path.name,
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
            "member_risk_records": str(member_records_path),
            "nonmember_risk_records": str(nonmember_records_path),
            "forget_member_index_files": forget_member_index_files,
            "matched_nonmember_index_files": matched_nonmember_index_files,
        },
        "surfaces": {
            "surface_a": {
                "name": surface_a["name"],
                "source_path": surface_a["source_path"],
                "source_kind": surface_a["source_kind"],
                "family": surface_a["family"],
                "member_orientation": orientation_a,
            },
            "surface_b": {
                "name": surface_b["name"],
                "source_path": surface_b["source_path"],
                "source_kind": surface_b["source_kind"],
                "family": surface_b["family"],
                "member_orientation": orientation_b,
            },
        },
        "alignment": aligned["alignment"],
        "runtime": {
            "weight_a": _round6(norm_weight_a),
            "weight_b": _round6(norm_weight_b),
            "top_fraction": _round6(top_fraction),
            "top_k_values": unique_top_k,
        },
        "member_risk": {
            "count": int(len(member_records)),
            "top_fraction_intersection_count": int(sum(1 for row in member_records if row["in_top_fraction_intersection"])),
            "top_rank_preview": [int(row["split_index"]) for row in member_records[: min(10, len(member_records))]],
        },
        "nonmember_risk": {
            "count": int(len(nonmember_records)),
            "top_fraction_intersection_count": int(
                sum(1 for row in nonmember_records if row["in_top_fraction_intersection"])
            ),
            "top_rank_preview": [int(row["split_index"]) for row in nonmember_records[: min(10, len(nonmember_records))]],
        },
        "forget_ladders": forget_ladders,
        "notes": [
            "This Step-0 surface aggregates aligned cross-box member-risk evidence into bounded forget-set exports.",
            "It does not claim unlearning success; it only makes the next 04-H1 pilot executable on current admitted assets.",
            "Matched nonmembers are selected by nearest combined-risk percentile for subset-board review.",
        ],
    }
    summary_path = workspace_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary


_SAMPLE_ID_PATTERN = re.compile(r"-(\d+)$")
DEFAULT_RISK_TARGETED_CHECKPOINT_ROOT = (
    Path(__file__).resolve().parents[3]
    / "workspaces"
    / "white-box"
    / "assets"
    / "gsa-cifar10-1k-3shadow-epoch300-rerun1"
    / "checkpoints"
    / "target"
)


def _extract_sample_id(path: str | Path) -> int:
    sample_path = Path(path)
    match = _SAMPLE_ID_PATTERN.search(sample_path.stem)
    if match is None:
        raise ValueError(f"Could not extract sample id from path: {sample_path}")
    return int(match.group(1))


def _scan_member_image_paths(member_dataset_dir: str | Path) -> dict[int, list[str]]:
    root = Path(member_dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Member dataset dir not found: {root}")
    image_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No image files found in member dataset dir: {root}")
    by_id: dict[int, list[str]] = {}
    for path in image_paths:
        sample_id = _extract_sample_id(path)
        by_id.setdefault(sample_id, []).append(str(path))
    return by_id


def _load_index_file(path: str | Path) -> list[int]:
    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Index file is empty: {path}")
    return [int(line.strip()) for line in raw.splitlines() if line.strip()]


def prepare_member_subsets(
    *,
    member_dataset_dir: str | Path,
    forget_member_index_file: str | Path,
    retain_max_samples: int | None = None,
    forget_max_samples: int | None = None,
) -> dict[str, Any]:
    image_by_id = _scan_member_image_paths(member_dataset_dir)
    forget_member_indices = _load_index_file(forget_member_index_file)
    missing = [int(index) for index in forget_member_indices if int(index) not in image_by_id]
    if missing:
        raise FileNotFoundError(
            f"Forget indices are missing from member dataset dir: {missing[:10]}"
        )

    forget_member_indices = sorted(dict.fromkeys(int(index) for index in forget_member_indices))
    if forget_max_samples is not None:
        forget_member_indices = forget_member_indices[: int(forget_max_samples)]
    retain_member_indices = sorted(index for index in image_by_id if index not in set(forget_member_indices))
    if retain_max_samples is not None:
        retain_member_indices = retain_member_indices[: int(retain_max_samples)]
    if not forget_member_indices:
        raise ValueError("Forget-member subset is empty after applying caps")
    if not retain_member_indices:
        raise ValueError("Retain-member subset is empty after applying caps")

    return {
        "member_dataset_dir": str(Path(member_dataset_dir)),
        "forget_member_indices": forget_member_indices,
        "forget_member_paths": [
            path
            for index in forget_member_indices
            for path in image_by_id[index]
        ],
        "retain_member_indices": retain_member_indices,
        "retain_member_paths": [
            path
            for index in retain_member_indices
            for path in image_by_id[index]
        ],
    }


def _build_ddpm_model(
    *,
    resolution: int = 32,
):
    from diffusers import UNet2DModel

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


def _resolve_checkpoint_dir(checkpoint_root: str | Path, checkpoint_dir: str | Path | None) -> Path:
    if checkpoint_dir is not None:
        resolved = Path(checkpoint_dir)
        if not resolved.exists():
            raise FileNotFoundError(f"Checkpoint dir not found: {resolved}")
        return resolved
    root = Path(checkpoint_root)
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {root}")
    candidates = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith("checkpoint-")
        ),
        key=lambda path: int(path.name.split("-", 1)[1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* dirs found under: {root}")
    return candidates[-1]


def _load_base_model(
    *,
    checkpoint_root: str | Path | None,
    checkpoint_dir: str | Path | None,
    device: str,
    resolution: int,
    random_init: bool,
):
    import torch

    model = _build_ddpm_model(resolution=resolution)
    resolved_checkpoint_dir: Path | None = None
    if not random_init:
        if checkpoint_root is None and checkpoint_dir is None:
            raise ValueError("A checkpoint root or checkpoint dir is required unless random_init=True")
        resolved_checkpoint_dir = _resolve_checkpoint_dir(
            checkpoint_root if checkpoint_root is not None else checkpoint_dir,
            checkpoint_dir,
        )
        weights_path = resolved_checkpoint_dir / "model.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(f"model.safetensors not found at {weights_path}")
        from safetensors.torch import load_file

        state_dict = load_file(str(weights_path), device="cpu")
        model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.train()
    return model, resolved_checkpoint_dir


def _load_image_batch(paths: list[str], *, resolution: int, device: str):
    import torch
    from PIL import Image

    batch: list[torch.Tensor] = []
    for raw_path in paths:
        image = Image.open(raw_path).convert("RGB")
        if image.size != (resolution, resolution):
            image = image.resize((resolution, resolution))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        batch.append(tensor.mul(2.0).sub(1.0))
    return torch.stack(batch, dim=0).to(device)


def _sample_paths(paths: list[str], batch_size: int, rng: random.Random) -> list[str]:
    if not paths:
        raise ValueError("Cannot sample from an empty path list")
    return [paths[rng.randrange(len(paths))] for _ in range(int(batch_size))]


def _set_training_seed(seed: int | None) -> None:
    import torch

    if seed is None:
        return
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def run_risk_targeted_unlearning_pilot(
    *,
    workspace: str | Path,
    member_dataset_dir: str | Path,
    forget_member_index_file: str | Path,
    checkpoint_root: str | Path | None = DEFAULT_RISK_TARGETED_CHECKPOINT_ROOT,
    checkpoint_dir: str | Path | None = None,
    matched_nonmember_index_file: str | Path | None = None,
    random_init: bool = False,
    retain_max_samples: int | None = None,
    forget_max_samples: int | None = None,
    num_steps: int = 100,
    batch_size: int = 4,
    num_workers: int = 0,
    lr: float = 1e-5,
    alpha: float = 0.5,
    mixture_lambda: float = 0.5,
    grad_clip: float = 1.0,
    resolution: int = 32,
    ddpm_num_train_timesteps: int = 1000,
    device: str = "cpu",
    seed: int | None = 0,
    provenance_status: str = "workspace-verified",
) -> dict[str, Any]:
    del num_workers
    import torch
    import torch.nn.functional as F
    from diffusers import DDPMScheduler
    from safetensors.torch import save_file

    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if float(lr) <= 0.0:
        raise ValueError("lr must be positive")
    if not 0.0 <= float(mixture_lambda) <= 1.0:
        raise ValueError("mixture_lambda must lie in [0, 1]")

    subsets = prepare_member_subsets(
        member_dataset_dir=member_dataset_dir,
        forget_member_index_file=forget_member_index_file,
        retain_max_samples=retain_max_samples,
        forget_max_samples=forget_max_samples,
    )
    _set_training_seed(seed)
    rng = random.Random(None if seed is None else int(seed) + 17)

    resolved_device = str(device)
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in the current runtime")

    model, resolved_checkpoint_dir = _load_base_model(
        checkpoint_root=checkpoint_root,
        checkpoint_dir=checkpoint_dir,
        device=resolved_device,
        resolution=resolution,
        random_init=random_init,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    scheduler = DDPMScheduler(
        num_train_timesteps=int(ddpm_num_train_timesteps),
        beta_schedule="linear",
        prediction_type="epsilon",
    )

    log_records: list[dict[str, Any]] = []
    branch_counts = {"keep_only": 0, "keep_minus_forget": 0}
    keep_losses: list[float] = []
    forget_losses: list[float] = []
    objective_values: list[float] = []

    for step in range(int(num_steps)):
        model.train()
        keep_paths = _sample_paths(subsets["retain_member_paths"], int(batch_size), rng)
        keep_batch = _load_image_batch(keep_paths, resolution=resolution, device=resolved_device)
        keep_timesteps = torch.randint(
            0,
            int(ddpm_num_train_timesteps),
            (keep_batch.shape[0],),
            device=keep_batch.device,
        ).long()
        keep_noise = torch.randn_like(keep_batch)
        noisy_keep = scheduler.add_noise(keep_batch, keep_noise, keep_timesteps)
        keep_pred = model(noisy_keep, timestep=keep_timesteps).sample
        keep_loss = F.mse_loss(keep_pred, keep_noise)

        include_forget = rng.random() < float(mixture_lambda)
        branch = "keep_only"
        objective = keep_loss
        forget_loss_value: float | None = None

        if include_forget:
            forget_paths = _sample_paths(subsets["forget_member_paths"], int(batch_size), rng)
            forget_batch = _load_image_batch(forget_paths, resolution=resolution, device=resolved_device)
            forget_timesteps = torch.randint(
                0,
                int(ddpm_num_train_timesteps),
                (forget_batch.shape[0],),
                device=forget_batch.device,
            ).long()
            forget_noise = torch.randn_like(forget_batch)
            noisy_forget = scheduler.add_noise(forget_batch, forget_noise, forget_timesteps)
            forget_pred = model(noisy_forget, timestep=forget_timesteps).sample
            forget_loss = F.mse_loss(forget_pred, forget_noise)
            objective = keep_loss - float(alpha) * forget_loss
            forget_loss_value = float(forget_loss.detach().cpu().item())
            branch = "keep_minus_forget"
            forget_losses.append(forget_loss_value)

        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        if float(grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()

        branch_counts[branch] += 1
        keep_loss_value = float(keep_loss.detach().cpu().item())
        objective_value = float(objective.detach().cpu().item())
        keep_losses.append(keep_loss_value)
        objective_values.append(objective_value)
        log_records.append(
            {
                "step": int(step),
                "branch": branch,
                "keep_loss": _round6(keep_loss_value),
                "forget_loss": _round6(forget_loss_value) if forget_loss_value is not None else None,
                "objective": _round6(objective_value),
            }
        )

    checkpoint_out_dir = workspace_path / "checkpoint-final"
    checkpoint_out_dir.mkdir(parents=True, exist_ok=True)
    model_cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    save_file(model_cpu_state, str(checkpoint_out_dir / "model.safetensors"))

    log_path = workspace_path / "train-log.jsonl"
    _write_jsonl(log_path, log_records)
    retain_index_path = workspace_path / "retain-members-used.txt"
    forget_index_path = workspace_path / "forget-members-used.txt"
    _write_index_file(retain_index_path, subsets["retain_member_indices"])
    _write_index_file(forget_index_path, subsets["forget_member_indices"])

    summary = {
        "status": "ready",
        "track": "defense",
        "method": "risk-targeted-siss-pilot",
        "paper": "Diffusion_Unlearning_SISS_ICLR2025",
        "mode": "risk-targeted-unlearning-pilot",
        "device": resolved_device,
        "gpu_release": "none",
        "admitted_change": "none",
        "provenance_status": provenance_status,
        "workspace": str(workspace_path),
        "workspace_name": workspace_path.name,
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
            "train_log": str(log_path),
            "checkpoint_dir": str(checkpoint_out_dir),
            "retain_member_index_file": str(retain_index_path),
            "forget_member_index_file": str(forget_index_path),
        },
        "base_model": {
            "random_init": bool(random_init),
            "checkpoint_root": str(checkpoint_root) if checkpoint_root is not None else None,
            "resolved_checkpoint_dir": str(resolved_checkpoint_dir) if resolved_checkpoint_dir else None,
        },
        "data": {
            "member_dataset_dir": str(member_dataset_dir),
            "matched_nonmember_index_file": str(matched_nonmember_index_file)
            if matched_nonmember_index_file is not None
            else None,
            "forget_member_count": int(len(subsets["forget_member_paths"])),
            "retain_member_count": int(len(subsets["retain_member_paths"])),
            "forget_member_unique_id_count": int(len(subsets["forget_member_indices"])),
            "retain_member_unique_id_count": int(len(subsets["retain_member_indices"])),
            "forget_member_preview": subsets["forget_member_indices"][: min(10, len(subsets["forget_member_indices"]))],
            "retain_member_preview": subsets["retain_member_indices"][: min(10, len(subsets["retain_member_indices"]))],
        },
        "objective": {
            "kind": "retain-plus-forget-hybrid",
            "alpha": _round6(alpha),
            "mixture_lambda": _round6(mixture_lambda),
            "grad_clip": _round6(grad_clip),
            "lr": _round6(lr),
        },
        "runtime": {
            "seed": int(seed) if seed is not None else None,
            "batch_size": int(batch_size),
            "num_steps": int(num_steps),
            "executed_steps": int(len(log_records)),
            "ddpm_num_train_timesteps": int(ddpm_num_train_timesteps),
            "resolution": int(resolution),
        },
        "training": {
            "branch_counts": branch_counts,
            "mean_keep_loss": _round6(np.mean(keep_losses)),
            "mean_objective": _round6(np.mean(objective_values)),
            "mean_forget_loss": _round6(np.mean(forget_losses)) if forget_losses else None,
            "tail_log": log_records[-min(10, len(log_records)) :],
        },
        "notes": [
            "This pilot only establishes execution of one bounded retain+forget training surface on current admitted assets.",
            "No attack-side rerun is attached yet, so this summary must not be read as a defense-positive verdict.",
        ],
    }
    summary_path = workspace_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary


def _load_shadow_reference_summary(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "ready" or payload.get("mode") != "loss-score-export":
        raise ValueError(f"Shadow reference summary must be a ready loss-score-export summary: {path}")
    return payload


def _build_subset_allowlist(
    *,
    forget_member_index_file: str | Path | None,
    matched_nonmember_index_file: str | Path | None,
) -> list[int] | None:
    if forget_member_index_file is None and matched_nonmember_index_file is None:
        return None
    if forget_member_index_file is None or matched_nonmember_index_file is None:
        raise ValueError("forget_member_index_file and matched_nonmember_index_file must be provided together")
    merged = sorted(
        {
            *(_load_index_file(forget_member_index_file)),
            *(_load_index_file(matched_nonmember_index_file)),
        }
    )
    if not merged:
        raise ValueError("Subset allowlist cannot be empty")
    return merged


def export_retained_highrisk_companion_subset(
    *,
    workspace: str | Path,
    member_risk_records_path: str | Path,
    nonmember_risk_records_path: str | Path,
    forget_member_index_file: str | Path,
    used_nonmember_index_file: str | Path,
    subset_size: int,
    provenance_status: str = "workspace-verified",
) -> dict[str, Any]:
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    if int(subset_size) <= 0:
        raise ValueError("subset_size must be positive")

    member_records = _read_jsonl_records(member_risk_records_path)
    nonmember_records = _read_jsonl_records(nonmember_risk_records_path)
    forget_member_ids = set(_load_index_file(forget_member_index_file))
    used_nonmember_ids = set(_load_index_file(used_nonmember_index_file))

    retained_member_records = [
        record
        for record in member_records
        if int(record["split_index"]) not in forget_member_ids
    ]
    available_nonmember_records = [
        record
        for record in nonmember_records
        if int(record["split_index"]) not in used_nonmember_ids
    ]
    if len(retained_member_records) < int(subset_size):
        raise ValueError("Not enough retained member records for requested subset_size")
    if len(available_nonmember_records) < int(subset_size):
        raise ValueError("Not enough available nonmember records for requested subset_size")

    selected_member_records = retained_member_records[: int(subset_size)]
    matched_nonmember_records = _match_nonmembers(selected_member_records, available_nonmember_records)
    member_indices = [int(record["split_index"]) for record in selected_member_records]
    matched_nonmember_indices = [int(record["split_index"]) for record in matched_nonmember_records]

    member_index_path = workspace_path / f"retained-highrisk-members-k{subset_size}.txt"
    nonmember_index_path = workspace_path / f"retained-highrisk-matched-nonmembers-k{subset_size}.txt"
    _write_index_file(member_index_path, member_indices)
    _write_index_file(nonmember_index_path, matched_nonmember_indices)

    summary = {
        "status": "ready",
        "track": "defense",
        "method": "risk-targeted-retained-companion",
        "mode": "retained-highrisk-companion-subset",
        "provenance_status": provenance_status,
        "workspace": str(workspace_path),
        "workspace_name": workspace_path.name,
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
            "member_index_file": str(member_index_path),
            "matched_nonmember_index_file": str(nonmember_index_path),
        },
        "subset_size": int(subset_size),
        "member_indices": member_indices,
        "matched_nonmember_indices": matched_nonmember_indices,
        "notes": [
            "This companion subset uses the highest-risk retained members after removing the exported forget list.",
            "Matched nonmembers exclude ids already consumed by the forgotten-subset board.",
        ],
    }
    summary_path = workspace_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary


def _write_review_packet_summary(
    *,
    output_path: Path,
    shadow_reference_summary: dict[str, Any],
    target_member_export: dict[str, Any],
    target_nonmember_export: dict[str, Any],
    label: str,
    sample_id_allowlist: list[int],
    checkpoint_dir: str | Path,
) -> None:
    packet_summary = {
        "status": "ready",
        "track": "white-box",
        "method": "gsa",
        "paper": "WhiteBox_GSA_PoPETS2025",
        "mode": "loss-score-export",
        "workspace": str(output_path.parent),
        "workspace_name": output_path.parent.name,
        "contract_stage": "target",
        "asset_grade": "real-asset-closed-loop",
        "provenance_status": "workspace-verified",
        "evidence_level": "bounded-loss-score-export",
        "artifact_paths": {
            "summary": str(output_path),
            "shadow_specs": shadow_reference_summary.get("artifact_paths", {}).get("shadow_specs", []),
        },
        "runtime": {
            **(shadow_reference_summary.get("runtime", {}) or {}),
            "subset_label": label,
            "sample_id_allowlist_count": int(len(sample_id_allowlist)) if sample_id_allowlist is not None else None,
            "review_checkpoint_dir": str(checkpoint_dir),
        },
        "exports": {
            "target_member": target_member_export,
            "target_non_member": target_nonmember_export,
            **{
                key: value
                for key, value in (shadow_reference_summary.get("exports", {}) or {}).items()
                if key.startswith("shadow_")
            },
        },
        "notes": [
            "Shadow exports are borrowed from an existing undefended packet for threshold-transfer diagnostics only.",
            "This packet is target-subset-only and must not be read as defense-aware.",
        ],
    }
    output_path.write_text(json.dumps(packet_summary, indent=2, ensure_ascii=True), encoding="utf-8")


def review_risk_targeted_unlearning_pilot(
    *,
    workspace: str | Path,
    shadow_reference_summary: str | Path,
    target_member_dataset_dir: str | Path,
    target_nonmember_dataset_dir: str | Path,
    baseline_checkpoint_root: str | Path,
    defended_checkpoint_dir: str | Path,
    forget_member_index_file: str | Path | None = None,
    matched_nonmember_index_file: str | Path | None = None,
    baseline_checkpoint_dir: str | Path | None = None,
    resolution: int = 32,
    ddpm_num_steps: int = 20,
    sampling_frequency: int = 2,
    attack_method: int = 1,
    prediction_type: str = "epsilon",
    device: str = "cpu",
    noise_seed: int | None = None,
    provenance_status: str = "workspace-verified",
) -> dict[str, Any]:
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    shadow_reference = _load_shadow_reference_summary(shadow_reference_summary)
    sample_id_allowlist = _build_subset_allowlist(
        forget_member_index_file=forget_member_index_file,
        matched_nonmember_index_file=matched_nonmember_index_file,
    )
    baseline_checkpoint_dir = (
        Path(baseline_checkpoint_dir)
        if baseline_checkpoint_dir is not None
        else _resolve_checkpoint_dir(baseline_checkpoint_root, None)
    )
    defended_checkpoint_dir = Path(defended_checkpoint_dir)

    export_root = workspace_path / "exports"
    export_root.mkdir(parents=True, exist_ok=True)

    baseline_member_export = _extract_gsa_loss_scores(
        dataset_dir=target_member_dataset_dir,
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
        sample_id_allowlist=sample_id_allowlist,
        noise_seed=noise_seed,
    )
    baseline_nonmember_export = _extract_gsa_loss_scores(
        dataset_dir=target_nonmember_dataset_dir,
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
        sample_id_allowlist=sample_id_allowlist,
        noise_seed=noise_seed,
    )
    defended_member_export = _extract_gsa_loss_scores(
        dataset_dir=target_member_dataset_dir,
        checkpoint_root=defended_checkpoint_dir.parent,
        checkpoint_dir=defended_checkpoint_dir,
        output_path=export_root / "defended-target-member-loss-scores.pt",
        records_path=export_root / "defended-target-member-loss-scores.jsonl",
        resolution=resolution,
        ddpm_num_steps=ddpm_num_steps,
        sampling_frequency=sampling_frequency,
        attack_method=attack_method,
        prediction_type=prediction_type,
        device=device,
        sample_id_allowlist=sample_id_allowlist,
        noise_seed=noise_seed,
    )
    defended_nonmember_export = _extract_gsa_loss_scores(
        dataset_dir=target_nonmember_dataset_dir,
        checkpoint_root=defended_checkpoint_dir.parent,
        checkpoint_dir=defended_checkpoint_dir,
        output_path=export_root / "defended-target-nonmember-loss-scores.pt",
        records_path=export_root / "defended-target-nonmember-loss-scores.jsonl",
        resolution=resolution,
        ddpm_num_steps=ddpm_num_steps,
        sampling_frequency=sampling_frequency,
        attack_method=attack_method,
        prediction_type=prediction_type,
        device=device,
        sample_id_allowlist=sample_id_allowlist,
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
        sample_id_allowlist=sample_id_allowlist,
        checkpoint_dir=baseline_checkpoint_dir,
    )
    _write_review_packet_summary(
        output_path=defended_packet_summary_path,
        shadow_reference_summary=shadow_reference,
        target_member_export=defended_member_export,
        target_nonmember_export=defended_nonmember_export,
        label="defended-target-subset",
        sample_id_allowlist=sample_id_allowlist,
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
    delta = {
        metric: _round6(float(defended_metrics[metric]) - float(baseline_metrics[metric]))
        for metric in metric_names
    }

    summary = {
        "status": "ready",
        "track": "defense",
        "method": "risk-targeted-siss-review",
        "paper": "Diffusion_Unlearning_SISS_ICLR2025",
        "mode": "risk-targeted-unlearning-review",
        "provenance_status": provenance_status,
        "workspace": str(workspace_path),
        "workspace_name": workspace_path.name,
        "artifact_paths": {
            "summary": str(workspace_path / "summary.json"),
            "baseline_packet_summary": str(baseline_packet_summary_path),
            "defended_packet_summary": str(defended_packet_summary_path),
            "baseline_threshold_eval_summary": str((workspace_path / "baseline-threshold-eval" / "summary.json")),
            "defended_threshold_eval_summary": str((workspace_path / "defended-threshold-eval" / "summary.json")),
        },
        "subset": {
            "forget_member_index_file": str(forget_member_index_file) if forget_member_index_file is not None else None,
            "matched_nonmember_index_file": str(matched_nonmember_index_file)
            if matched_nonmember_index_file is not None
            else None,
            "sample_id_allowlist": sample_id_allowlist,
            "sample_id_allowlist_count": int(len(sample_id_allowlist)) if sample_id_allowlist is not None else None,
            "noise_seed": int(noise_seed) if noise_seed is not None else None,
        },
        "baseline": {
            "checkpoint_dir": str(baseline_checkpoint_dir),
            "target_member_export": baseline_member_export,
            "target_nonmember_export": baseline_nonmember_export,
            "threshold_eval": baseline_eval,
        },
        "defended": {
            "checkpoint_dir": str(defended_checkpoint_dir),
            "target_member_export": defended_member_export,
            "target_nonmember_export": defended_nonmember_export,
            "threshold_eval": defended_eval,
        },
        "comparison": {
            "target_transfer_delta": delta,
        },
        "notes": [
            "This review reuses an existing undefended shadow export only for threshold-transfer diagnostics.",
            "It is attack-side readable, but not defense-aware; any positive or negative read remains provisional until defended shadows are retrained.",
        ],
    }
    summary_path = workspace_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary
