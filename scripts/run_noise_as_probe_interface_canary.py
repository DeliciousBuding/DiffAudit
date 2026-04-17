from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import BlipForConditionalGeneration, BlipProcessor

from diffusers import (
    AutoencoderKL,
    DDIMInverseScheduler,
    DDIMScheduler,
    StableDiffusionPipeline,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external.SecMI.mia_evals.measures.ssim import ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded Noise-as-a-Probe interface canary.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--member-dir", type=Path, required=True)
    parser.add_argument("--nonmember-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--lora-dir", type=Path, required=True)
    parser.add_argument("--blip-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--inversion-steps", type=int, default=50)
    parser.add_argument("--generation-steps", type=int, default=50)
    parser.add_argument("--inversion-guidance-scale", type=float, default=1.0)
    parser.add_argument("--generation-guidance-scale", type=float, default=3.5)
    parser.add_argument(
        "--generation-guidance-mode",
        choices=["fixed", "hidden-jitter"],
        default="fixed",
    )
    parser.add_argument(
        "--generation-guidance-scales",
        type=str,
        default=None,
        help="Comma-separated candidate guidance scales for fixed or hidden-jitter execution.",
    )
    parser.add_argument(
        "--generation-guidance-seed",
        type=int,
        default=0,
        help="Deterministic seed for per-sample hidden guidance selection.",
    )
    parser.add_argument("--distance-metric", choices=["mse", "ssim"], default="mse")
    parser.add_argument("--member-limit", type=int, default=1)
    parser.add_argument("--member-offset", type=int, default=0)
    parser.add_argument("--nonmember-limit", type=int, default=1)
    parser.add_argument("--nonmember-offset", type=int, default=0)
    parser.add_argument("--calibration-limit", type=int, default=0)
    parser.add_argument("--calibration-offset", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_rows(split_dir: Path, subset_limit: int, offset: int = 0) -> list[dict]:
    if subset_limit <= 0:
        return []
    rows: list[dict] = []
    metadata_path = split_dir / "metadata.jsonl"
    loaded = 0
    for raw_line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        if loaded < offset:
            loaded += 1
            continue
        rows.append(json.loads(raw_line))
        loaded += 1
        if len(rows) >= subset_limit:
            break
    return rows


def build_transform(resolution: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def parse_generation_guidance_scales(raw_value: str | None, fallback_scale: float) -> list[float]:
    if raw_value is None or not raw_value.strip():
        return [float(fallback_scale)]
    scales: list[float] = []
    for token in raw_value.split(","):
        value = token.strip()
        if not value:
            continue
        scales.append(float(value))
    if not scales:
        raise ValueError("generation-guidance-scales must contain at least one numeric value.")
    return scales


def resolve_generation_guidance_config(args: argparse.Namespace) -> dict[str, object]:
    scales = parse_generation_guidance_scales(
        getattr(args, "generation_guidance_scales", None),
        float(args.generation_guidance_scale),
    )
    mode = str(args.generation_guidance_mode)
    if mode == "fixed":
        if len(scales) != 1:
            raise ValueError("fixed generation-guidance-mode requires exactly one guidance scale.")
        return {
            "mode": mode,
            "scales": [float(scales[0])],
            "seed": int(args.generation_guidance_seed),
            "selection_rule": "fixed guidance_scale for every sample",
        }
    unique_scales = list(dict.fromkeys(float(scale) for scale in scales))
    if len(unique_scales) < 2:
        raise ValueError("hidden-jitter generation-guidance-mode requires at least two candidate scales.")
    return {
        "mode": mode,
        "scales": unique_scales,
        "seed": int(args.generation_guidance_seed),
        "selection_rule": (
            "deterministic hidden per-sample choice keyed by generation_guidance_seed + split + file_name"
        ),
    }


def deterministic_guidance_choice(
    *,
    split_name: str,
    file_name: str,
    scales: list[float],
    seed: int,
) -> tuple[int, float]:
    key = f"{seed}:{split_name}:{file_name}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    index = int.from_bytes(digest[:8], "big") % len(scales)
    return index, float(scales[index])


def normalize_record_split(split_name: str) -> str:
    if split_name.startswith("calibration_nonmember"):
        return "calibration_nonmember"
    if split_name.startswith("nonmember"):
        return "nonmember"
    if split_name.startswith("member"):
        return "member"
    return split_name


def summarize_guidance_usage(records: list[dict]) -> dict[str, dict[str, int]]:
    usage: dict[str, dict[str, int]] = {"all": {}}
    for record in records:
        scale_key = str(float(record["generation_guidance_scale_used"]))
        usage["all"][scale_key] = usage["all"].get(scale_key, 0) + 1
        split_key = normalize_record_split(str(record["split"]))
        split_usage = usage.setdefault(split_key, {})
        split_usage[scale_key] = split_usage.get(scale_key, 0) + 1
    return usage


def ensure_disjoint_rows(
    primary_rows: list[dict],
    calibration_rows: list[dict],
    *,
    primary_name: str,
    calibration_name: str,
) -> None:
    primary_files = {str(row["file_name"]) for row in primary_rows}
    calibration_files = {str(row["file_name"]) for row in calibration_rows}
    overlap = sorted(primary_files & calibration_files)
    if overlap:
        preview = ", ".join(overlap[:5])
        raise ValueError(
            f"{primary_name} and {calibration_name} must be disjoint by file_name; overlap: {preview}"
        )


def tensor_to_uint8_image(pixel: torch.Tensor) -> Image.Image:
    image_tensor = ((pixel.detach().float().cpu().squeeze(0) + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    image_tensor = image_tensor.permute(1, 2, 0).contiguous()
    return Image.fromarray(image_tensor.numpy(), mode="RGB")


class NoiseAsProbeCanaryRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.transform = build_transform(args.resolution)
        self.guidance_config = resolve_generation_guidance_config(args)
        self._blip_processor: BlipProcessor | None = None
        self._blip_model: BlipForConditionalGeneration | None = None
        self._load_models()

    def _load_models(self) -> None:
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.args.model_dir,
            torch_dtype=self.dtype,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            local_files_only=True,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.unet.load_attn_procs(self.args.lora_dir.as_posix())
        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        self.vae = AutoencoderKL.from_pretrained(
            self.args.model_dir,
            subfolder="vae",
            torch_dtype=self.dtype,
            local_files_only=True,
        ).to(self.device)
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.unet = self.pipe.unet
        self.inverse_scheduler = DDIMInverseScheduler.from_config(self.pipe.scheduler.config)
        self.forward_scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        self.vae.eval()
        self.text_encoder.eval()
        self.unet.eval()

    def _ensure_blip(self) -> None:
        if self._blip_processor is not None and self._blip_model is not None:
            return
        self._blip_processor = BlipProcessor.from_pretrained(self.args.blip_dir, local_files_only=True)
        self._blip_model = BlipForConditionalGeneration.from_pretrained(
            self.args.blip_dir, torch_dtype=self.dtype, local_files_only=True
        ).to(self.device)
        self._blip_model.eval()

    def prompt_for(self, row: dict, image: Image.Image) -> tuple[str, str]:
        caption = row.get("text")
        if isinstance(caption, str) and caption.strip():
            return caption.strip(), "metadata"
        self._ensure_blip()
        assert self._blip_processor is not None
        assert self._blip_model is not None
        inputs = self._blip_processor(images=image, return_tensors="pt").to(self.device, self.dtype)
        with torch.inference_mode():
            generated = self._blip_model.generate(**inputs, max_new_tokens=32)
        prompt = self._blip_processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return prompt, "blip"

    def encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        cond_inputs = self.tokenizer(
            [prompt],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_inputs = self.tokenizer(
            [""],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            cond = self.text_encoder(cond_inputs.input_ids.to(self.device))[0]
            uncond = self.text_encoder(uncond_inputs.input_ids.to(self.device))[0]
        return cond, uncond

    def encode_image(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        pixel = self.transform(image).unsqueeze(0).to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            latent_dist = self.vae.encode(pixel).latent_dist
            latent = latent_dist.mean * self.vae.config.scaling_factor
        return pixel, latent

    def predict_noise(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        cond_embeds: torch.Tensor,
        uncond_embeds: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        latent_pair = torch.cat([latent, latent], dim=0)
        embeds = torch.cat([uncond_embeds, cond_embeds], dim=0).to(dtype=self.dtype)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"
        ):
            noise_pair = self.unet(latent_pair, timestep.repeat(2), encoder_hidden_states=embeds).sample
        noise_uncond, noise_cond = noise_pair.chunk(2)
        return noise_uncond + guidance_scale * (noise_cond - noise_uncond)

    def invert_latent(
        self, latent: torch.Tensor, cond_embeds: torch.Tensor, uncond_embeds: torch.Tensor
    ) -> torch.Tensor:
        self.inverse_scheduler.set_timesteps(self.args.inversion_steps, device=self.device)
        current = latent.clone()
        for timestep in self.inverse_scheduler.timesteps:
            timestep_tensor = torch.tensor([int(timestep)], device=self.device, dtype=torch.long)
            noise_pred = self.predict_noise(
                current,
                timestep_tensor,
                cond_embeds,
                uncond_embeds,
                self.args.inversion_guidance_scale,
            )
            step_output = self.inverse_scheduler.step(noise_pred, int(timestep), current)
            current = step_output.prev_sample
        return current

    def generation_guidance_for(self, split_name: str, row: dict) -> tuple[int, float]:
        scales = list(self.guidance_config["scales"])
        if str(self.guidance_config["mode"]) == "fixed":
            return 0, float(scales[0])
        return deterministic_guidance_choice(
            split_name=split_name,
            file_name=str(row["file_name"]),
            scales=scales,
            seed=int(self.guidance_config["seed"]),
        )

    def generate_from_latents(self, prompt: str, latents: torch.Tensor, guidance_scale: float) -> Image.Image:
        self.pipe.scheduler = self.forward_scheduler
        self.pipe.scheduler.set_timesteps(self.args.generation_steps, device=self.device)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"
        ):
            result = self.pipe(
                prompt=prompt,
                negative_prompt="",
                num_inference_steps=self.args.generation_steps,
                guidance_scale=guidance_scale,
                height=self.args.resolution,
                width=self.args.resolution,
                latents=latents.clone().to(device=self.device, dtype=self.dtype),
                output_type="pil",
            )
        return result.images[0].convert("RGB")

    def distance_metrics(self, query_pixel: torch.Tensor, replay_image: Image.Image) -> dict[str, float]:
        replay_pixel = self.transform(replay_image).unsqueeze(0).to(self.device, dtype=self.dtype)
        query_255 = ((query_pixel.float() + 1.0) * 127.5).clamp(0, 255)
        replay_255 = ((replay_pixel.float() + 1.0) * 127.5).clamp(0, 255)
        mse = float(torch.mean((query_255 - replay_255) ** 2).detach().cpu())
        mae = float(torch.mean(torch.abs(query_255 - replay_255)).detach().cpu())
        ssim_value = float(ssim(query_255, replay_255, data_range=255, size_average=True).detach().cpu())
        psnr = 99.0 if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
        primary = mse if self.args.distance_metric == "mse" else 1.0 - ssim_value
        return {
            "primary_distance": float(primary),
            "mse": mse,
            "mae": mae,
            "ssim": ssim_value,
            "psnr": float(psnr),
        }

    def canary_row(
        self,
        split_name: str,
        split_dir: Path,
        row: dict,
        output_dir: Path,
    ) -> dict:
        image = Image.open(split_dir / row["file_name"]).convert("RGB")
        prompt, prompt_source = self.prompt_for(row, image)
        query_pixel, latent = self.encode_image(image)
        cond_embeds, uncond_embeds = self.encode_prompt(prompt)
        inverted_latent = self.invert_latent(latent, cond_embeds, uncond_embeds)
        guidance_choice_index, guidance_scale = self.generation_guidance_for(split_name, row)
        replay_image = self.generate_from_latents(prompt, inverted_latent, guidance_scale=guidance_scale)
        metrics = self.distance_metrics(query_pixel, replay_image)

        stem = f"{split_name}_{Path(row['file_name']).stem}"
        query_path = output_dir / f"{stem}_query.png"
        replay_path = output_dir / f"{stem}_replay.png"
        latent_path = output_dir / f"{stem}_inverted_latent.pt"
        tensor_to_uint8_image(query_pixel).save(query_path)
        replay_image.save(replay_path)
        torch.save(inverted_latent.detach().float().cpu(), latent_path)

        return {
            "split": split_name,
            "file_name": row["file_name"],
            "prompt": prompt,
            "prompt_source": prompt_source,
            "generation_guidance_mode": str(self.guidance_config["mode"]),
            "generation_guidance_scale_used": float(guidance_scale),
            "generation_guidance_choice_index": int(guidance_choice_index),
            "generation_guidance_choice_key": f"{split_name}:{row['file_name']}",
            "query_path": query_path.as_posix(),
            "replay_path": replay_path.as_posix(),
            "inverted_latent_path": latent_path.as_posix(),
            **metrics,
        }


def threshold_from_calibration(records: list[dict]) -> float | None:
    if not records:
        return None
    scores = np.asarray([float(item["primary_distance"]) for item in records], dtype=float)
    return float(np.percentile(scores, 15.0))


def evaluate_threshold(
    member_records: list[dict],
    nonmember_records: list[dict],
    threshold: float | None,
) -> dict[str, float] | None:
    if threshold is None:
        return None
    member_scores = np.asarray([float(item["primary_distance"]) for item in member_records], dtype=float)
    nonmember_scores = np.asarray([float(item["primary_distance"]) for item in nonmember_records], dtype=float)
    tp = float((member_scores <= threshold).sum())
    fn = float((member_scores > threshold).sum())
    fp = float((nonmember_scores <= threshold).sum())
    tn = float((nonmember_scores > threshold).sum())
    total = tp + fn + fp + tn
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "accuracy": float((tp + tn) / total) if total else 0.0,
        "tpr": float(tp / (tp + fn)) if tp + fn else 0.0,
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
    }


def main() -> int:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    output_dir = args.run_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    member_rows = load_rows(args.member_dir, subset_limit=args.member_limit, offset=args.member_offset)
    nonmember_rows = load_rows(args.nonmember_dir, subset_limit=args.nonmember_limit, offset=args.nonmember_offset)
    calibration_rows = load_rows(
        args.nonmember_dir,
        subset_limit=args.calibration_limit,
        offset=args.calibration_offset,
    )
    if not member_rows or not nonmember_rows:
        raise ValueError("Both member-dir and nonmember-dir need at least one metadata row.")
    ensure_disjoint_rows(
        nonmember_rows,
        calibration_rows,
        primary_name="evaluation nonmember rows",
        calibration_name="calibration rows",
    )

    runner = NoiseAsProbeCanaryRunner(args)
    member_records = [
        runner.canary_row(f"member_{idx:03d}", args.member_dir, row, output_dir)
        for idx, row in enumerate(member_rows)
    ]
    nonmember_records = [
        runner.canary_row(f"nonmember_{idx:03d}", args.nonmember_dir, row, output_dir)
        for idx, row in enumerate(nonmember_rows)
    ]
    calibration_records = [
        runner.canary_row(f"calibration_nonmember_{idx:03d}", args.nonmember_dir, row, output_dir)
        for idx, row in enumerate(calibration_rows)
    ]
    threshold = threshold_from_calibration(calibration_records)
    threshold_eval = evaluate_threshold(member_records, nonmember_records, threshold)
    records = member_records + nonmember_records
    all_records = records + calibration_records

    summary = {
        "status": "interface-canary-only" if args.member_limit == 1 and args.nonmember_limit == 1 else "bounded-expansion-rung",
        "family": "noise-as-a-probe",
        "target_family": "sd15-plus-celeba_partial_target-checkpoint-25000",
        "first_smoke_contract": "one-member-one-nonmember-interface-canary",
        "distance_metric": args.distance_metric,
        "distance_direction": "lower-is-more-member-like",
        "inversion_steps": args.inversion_steps,
        "generation_steps": args.generation_steps,
        "generation_guidance_scale": (
            float(runner.guidance_config["scales"][0]) if runner.guidance_config["mode"] == "fixed" else None
        ),
        "generation_guidance_mode": runner.guidance_config["mode"],
        "generation_guidance_scales": runner.guidance_config["scales"],
        "generation_guidance_seed": runner.guidance_config["seed"],
        "generation_guidance_selection_rule": runner.guidance_config["selection_rule"],
        "inversion_guidance_scale": args.inversion_guidance_scale,
        "member_limit": args.member_limit,
        "nonmember_limit": args.nonmember_limit,
        "calibration_limit": args.calibration_limit,
        "gpu_release": "none",
        "records": records,
        "calibration_records": calibration_records,
        "guidance_usage": summarize_guidance_usage(all_records),
        "threshold_candidate": threshold_eval,
        "duration_seconds": round(time.perf_counter() - start_time, 3),
    }
    (args.run_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.run_root / "records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    if calibration_records:
        (args.run_root / "calibration_records.json").write_text(
            json.dumps(calibration_records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
