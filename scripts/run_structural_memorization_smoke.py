from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torchvision import transforms
from transformers import BlipForConditionalGeneration, BlipProcessor, CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external.SecMI.mia_evals.measures.ssim import ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded structural-memorization gray-box smoke.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--member-dir", type=Path, required=True)
    parser.add_argument("--nonmember-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--lora-dir", type=Path, required=True)
    parser.add_argument("--blip-dir", type=Path, required=True)
    parser.add_argument("--subset-limit", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--total-timestep", type=int, default=100)
    parser.add_argument("--interval", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_rows(split_dir: Path, subset_limit: int) -> list[dict]:
    rows = []
    metadata_path = split_dir / "metadata.jsonl"
    for raw_line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        rows.append(json.loads(raw_line))
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


def search_threshold(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> dict[str, float]:
    min_value = float(min(member_scores.min(), nonmember_scores.min()))
    max_value = float(max(member_scores.max(), nonmember_scores.max()))
    best_threshold = min_value
    best_asr = -1.0
    best_tpr = 0.0
    best_fpr = 1.0
    labels = np.concatenate([np.ones_like(member_scores), np.zeros_like(nonmember_scores)])
    scores = np.concatenate([member_scores, nonmember_scores])
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    tpr_at_1pct_fpr = 0.0
    for candidate_fpr, candidate_tpr in zip(fpr, tpr):
        if candidate_fpr >= 0.01:
            tpr_at_1pct_fpr = float(candidate_tpr)
            break
    for threshold in np.linspace(min_value, max_value, num=2000, endpoint=True):
        tp = float((member_scores >= threshold).sum())
        fn = float((member_scores < threshold).sum())
        fp = float((nonmember_scores >= threshold).sum())
        tn = float((nonmember_scores < threshold).sum())
        asr = (tp + tn) / (tp + tn + fp + fn)
        if asr > best_asr:
            best_asr = float(asr)
            best_threshold = float(threshold)
            best_tpr = tp / (tp + fn) if tp + fn else 0.0
            best_fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "auc": auc,
        "asr": best_asr,
        "threshold": best_threshold,
        "tpr_at_1pct_fpr": tpr_at_1pct_fpr,
        "best_tpr": float(best_tpr),
        "best_fpr": float(best_fpr),
    }


class StructuralMemorizationRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.transform = build_transform(args.resolution)
        self._blip_processor: BlipProcessor | None = None
        self._blip_model: BlipForConditionalGeneration | None = None
        self._load_models()

    def _load_models(self) -> None:
        self.vae = AutoencoderKL.from_pretrained(
            self.args.model_dir, subfolder="vae", torch_dtype=self.dtype, local_files_only=True
        ).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.args.model_dir, subfolder="tokenizer", local_files_only=True
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.args.model_dir, subfolder="text_encoder", torch_dtype=self.dtype, local_files_only=True
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            self.args.model_dir, subfolder="unet", torch_dtype=self.dtype, local_files_only=True
        ).to(self.device)
        if self.args.lora_dir.is_dir():
            self.unet.load_attn_procs(self.args.lora_dir.as_posix())
        self.scheduler = DDIMScheduler.from_pretrained(
            self.args.model_dir, subfolder="scheduler", local_files_only=True
        )
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

    def caption_for_image(self, row: dict, image: Image.Image) -> str:
        caption = row.get("text")
        if isinstance(caption, str) and caption.strip():
            return caption.strip()
        self._ensure_blip()
        assert self._blip_processor is not None
        assert self._blip_model is not None
        inputs = self._blip_processor(images=image, return_tensors="pt").to(self.device, self.dtype)
        with torch.inference_mode():
            generated = self._blip_model.generate(**inputs, max_new_tokens=32)
        return self._blip_processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

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

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            decoded = self.vae.decode(latent / self.vae.config.scaling_factor).sample
        return decoded.clamp(-1.0, 1.0)

    def guided_noise_pred(
        self, latent: torch.Tensor, timestep: int, cond_embeds: torch.Tensor, uncond_embeds: torch.Tensor
    ) -> torch.Tensor:
        timestep_tensor = torch.tensor([timestep], device=self.device, dtype=torch.long)
        latent_pair = torch.cat([latent, latent], dim=0)
        embeds = torch.cat([uncond_embeds, cond_embeds], dim=0).to(dtype=self.dtype)
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            noise_pair = self.unet(latent_pair, timestep_tensor.repeat(2), embeds).sample
        noise_uncond, noise_cond = noise_pair.chunk(2)
        return noise_uncond + self.args.guidance_scale * (noise_cond - noise_uncond)

    def forward_corrupt(
        self, latent: torch.Tensor, cond_embeds: torch.Tensor, uncond_embeds: torch.Tensor
    ) -> tuple[torch.Tensor, list[int]]:
        schedule = list(range(0, self.args.total_timestep + 1, self.args.interval))
        if schedule[-1] != self.args.total_timestep:
            schedule.append(self.args.total_timestep)
        current = latent
        for current_t, next_t in zip(schedule[:-1], schedule[1:]):
            alpha_t = self.scheduler.alphas_cumprod[current_t].to(self.device, dtype=self.dtype)
            alpha_next = self.scheduler.alphas_cumprod[next_t].to(self.device, dtype=self.dtype)
            noise_pred = self.guided_noise_pred(current, current_t, cond_embeds, uncond_embeds)
            pred_x0 = (current - torch.sqrt(1.0 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            current = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1.0 - alpha_next) * noise_pred
        return current, schedule

    def compute_structural_score(self, split_dir: Path, row: dict) -> dict:
        image = Image.open(split_dir / row["file_name"]).convert("RGB")
        prompt = self.caption_for_image(row, image)
        pixel, latent = self.encode_image(image)
        cond_embeds, uncond_embeds = self.encode_prompt(prompt)
        corrupted_latent, schedule = self.forward_corrupt(latent, cond_embeds, uncond_embeds)
        reconstructed = self.decode_latent(corrupted_latent)
        pixel_255 = ((pixel.float() + 1.0) * 127.5).clamp(0, 255)
        reconstructed_255 = ((reconstructed.float() + 1.0) * 127.5).clamp(0, 255)
        score = float(ssim(pixel_255, reconstructed_255, data_range=255, size_average=True).detach().cpu())
        mae = float(torch.mean(torch.abs(pixel_255 - reconstructed_255)).detach().cpu())
        mse = float(torch.mean((pixel_255 - reconstructed_255) ** 2).detach().cpu())
        psnr = 99.0 if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
        return {
            "file_name": row["file_name"],
            "prompt": prompt,
            "score": score,
            "mae": mae,
            "psnr": psnr,
            "schedule": schedule,
        }


def run_split(
    runner: StructuralMemorizationRunner, split_name: str, split_dir: Path, rows: list[dict], output_dir: Path
) -> list[dict]:
    records = []
    for index, row in enumerate(rows, start=1):
        result = runner.compute_structural_score(split_dir, row)
        result["split"] = split_name
        result["index"] = index
        records.append(result)
        print(f"[{split_name}] {index}/{len(rows)} {row['file_name']} score={result['score']:.6f} psnr={result['psnr']:.3f}")
    (output_dir / f"{split_name}_scores.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / f"{split_name}_scores.txt").write_text(
        "\n".join(f"{item['file_name']}\t{item['score']:.8f}\t{item['psnr']:.5f}\t{item['mae']:.5f}" for item in records),
        encoding="utf-8",
    )
    return records


def main() -> None:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    output_dir = args.run_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    member_rows = load_rows(args.member_dir, args.subset_limit)
    nonmember_rows = load_rows(args.nonmember_dir, args.subset_limit)
    start = time.time()
    runner = StructuralMemorizationRunner(args)
    member_records = run_split(runner, "member", args.member_dir, member_rows, output_dir)
    nonmember_records = run_split(runner, "nonmember", args.nonmember_dir, nonmember_rows, output_dir)
    elapsed_hours = (time.time() - start) / 3600.0

    member_scores = np.asarray([item["score"] for item in member_records], dtype=float)
    nonmember_scores = np.asarray([item["score"] for item in nonmember_records], dtype=float)
    metrics = search_threshold(member_scores, nonmember_scores)
    status = "candidate" if metrics["auc"] >= 0.55 else "negative but useful"

    config = {
        "run_id": args.run_root.name,
        "track": "gray-box",
        "method": "structural-memorization",
        "status": "executed",
        "mode": "local-faithful-approximation-smoke",
        "assets": {
            "member_dataset": args.member_dir.as_posix(),
            "nonmember_dataset": args.nonmember_dir.as_posix(),
            "model_dir": args.model_dir.as_posix(),
            "lora_dir": args.lora_dir.as_posix(),
            "blip_dir": args.blip_dir.as_posix(),
        },
        "parameters": {
            "subset_limit": args.subset_limit,
            "resolution": args.resolution,
            "guidance_scale": args.guidance_scale,
            "total_timestep": args.total_timestep,
            "interval": args.interval,
            "caption_source": "metadata-with-blip-fallback",
        },
    }
    (args.run_root / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "run_id": args.run_root.name,
        "track": "gray-box",
        "method": "structural-memorization",
        "dataset": "CelebA",
        "model": "StableDiffusion-v1.5 + Recon target LoRA",
        "mode": "local-faithful-approximation-smoke",
        "status": status,
        "metrics": {
            "auc": round(float(metrics["auc"]), 6),
            "asr": round(float(metrics["asr"]), 6),
            "tpr_at_1pct_fpr": round(float(metrics["tpr_at_1pct_fpr"]), 6),
        },
        "analysis": {
            "threshold": round(float(metrics["threshold"]), 6),
            "best_tpr": round(float(metrics["best_tpr"]), 6),
            "best_fpr": round(float(metrics["best_fpr"]), 6),
            "member_score_mean": round(float(member_scores.mean()), 6),
            "nonmember_score_mean": round(float(nonmember_scores.mean()), 6),
            "member_psnr_mean": round(float(np.mean([item["psnr"] for item in member_records])), 6),
            "nonmember_psnr_mean": round(float(np.mean([item["psnr"] for item in nonmember_records])), 6),
            "member_rows": len(member_records),
            "nonmember_rows": len(nonmember_records),
            "schedule": member_records[0]["schedule"] if member_records else [],
        },
        "cost": {
            "gpu_hours": round(float(elapsed_hours), 4),
        },
        "notes": (
            "Bounded structural-memorization smoke on the local CelebA target-family stack. "
            "This is a faithful local approximation using cached metadata captions with BLIP fallback, "
            "bounded DDIM-style forward corruption, and image-space SSIM scoring."
        ),
        "artifacts": {
            "member_scores": (output_dir / "member_scores.txt").as_posix(),
            "nonmember_scores": (output_dir / "nonmember_scores.txt").as_posix(),
        },
    }
    (args.run_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
