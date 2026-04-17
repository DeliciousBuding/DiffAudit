from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from transformers import (
    AutoProcessor,
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPModel,
)

from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external.SecMI.mia_evals.measures.ssim import ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded black-box semantic auxiliary-classifier probe.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--member-dir", type=Path, required=True)
    parser.add_argument("--nonmember-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--lora-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--blip-dir", type=Path, required=True)
    parser.add_argument("--subset-limit", type=int, default=8)
    parser.add_argument("--num-returns", type=int, default=3)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260415)
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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def search_threshold(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    best_asr = -1.0
    best_threshold = float(scores.min())
    best_tpr = 0.0
    best_fpr = 1.0
    tpr_at_1pct_fpr = 0.0
    for candidate_fpr, candidate_tpr in zip(fpr, tpr):
        if candidate_fpr >= 0.01:
            tpr_at_1pct_fpr = float(candidate_tpr)
            break
    for threshold in np.linspace(float(scores.min()), float(scores.max()), num=2000, endpoint=True):
        preds = (scores >= threshold).astype(np.int64)
        tp = float(((preds == 1) & (labels == 1)).sum())
        tn = float(((preds == 0) & (labels == 0)).sum())
        fp = float(((preds == 1) & (labels == 0)).sum())
        fn = float(((preds == 0) & (labels == 1)).sum())
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


class ProbeRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
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
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.unet.load_attn_procs(self.args.lora_dir.as_posix())
        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.clip_processor = AutoProcessor.from_pretrained(self.args.clip_dir, local_files_only=True)
        self.clip_model = CLIPModel.from_pretrained(
            self.args.clip_dir, torch_dtype=self.dtype, local_files_only=True
        ).to(self.device)
        self.clip_model.eval()

    def _ensure_blip(self) -> None:
        if self._blip_processor is not None and self._blip_model is not None:
            return
        self._blip_processor = BlipProcessor.from_pretrained(self.args.blip_dir, local_files_only=True)
        self._blip_model = BlipForConditionalGeneration.from_pretrained(
            self.args.blip_dir, torch_dtype=self.dtype, local_files_only=True
        ).to(self.device)
        self._blip_model.eval()

    def prompt_for(self, row: dict, image: Image.Image) -> str:
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

    def image_embedding(self, image: Image.Image) -> np.ndarray:
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            vision_outputs = self.clip_model.vision_model(pixel_values=inputs["pixel_values"])
            features = vision_outputs.pooler_output
        features = features / features.norm(dim=-1, keepdim=True)
        return features[0].detach().float().cpu().numpy()

    def generate_returns(self, prompt: str, base_seed: int) -> list[Image.Image]:
        images = []
        for idx in range(self.args.num_returns):
            generator = torch.Generator(device=self.device.type)
            generator.manual_seed(base_seed + idx)
            with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
                result = self.pipe(
                    prompt=prompt,
                    num_inference_steps=self.args.num_inference_steps,
                    guidance_scale=self.args.guidance_scale,
                    height=self.args.resolution,
                    width=self.args.resolution,
                    generator=generator,
                )
            images.append(result.images[0].convert("RGB"))
        return images

    def ssim_score(self, query: Image.Image, generated: Image.Image) -> float:
        q = torch.from_numpy(np.asarray(query, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
        g = torch.from_numpy(np.asarray(generated, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
        return float(ssim(q, g, data_range=255, size_average=True).detach().cpu())

    def feature_row(self, split_dir: Path, row: dict, base_seed: int, output_dir: Path, sample_index: int) -> dict:
        query_image = Image.open(split_dir / row["file_name"]).convert("RGB").resize((self.args.resolution, self.args.resolution))
        prompt = self.prompt_for(row, query_image)
        query_embed = self.image_embedding(query_image)
        returns = self.generate_returns(prompt, base_seed)
        return_embeds = [self.image_embedding(image) for image in returns]
        cosines = np.asarray([cosine_similarity(query_embed, embed) for embed in return_embeds], dtype=float)
        ssim_values = np.asarray([self.ssim_score(query_image, image) for image in returns], dtype=float)
        mean_cos = float(cosines.mean())
        max_cos = float(cosines.max())
        std_cos = float(cosines.std())
        gap_cos = float(max_cos - mean_cos)
        mean_ssim = float(ssim_values.mean())
        max_ssim = float(ssim_values.max())
        output_prefix = f"{sample_index:03d}_{Path(row['file_name']).stem}"
        for idx, image in enumerate(returns):
            image.save(output_dir / f"{output_prefix}_ret{idx}.png")
        return {
            "file_name": row["file_name"],
            "prompt": prompt,
            "mean_cos": mean_cos,
            "max_cos": max_cos,
            "std_cos": std_cos,
            "gap_cos": gap_cos,
            "mean_ssim": mean_ssim,
            "max_ssim": max_ssim,
        }


def main() -> None:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    output_dir = args.run_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_dir = output_dir / "generated"
    generation_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    member_rows = load_rows(args.member_dir, args.subset_limit)
    nonmember_rows = load_rows(args.nonmember_dir, args.subset_limit)

    config = {
        "run_id": args.run_root.name,
        "track": "black-box",
        "method": "semantic-auxiliary-classifier",
        "status": "running",
        "mode": "small-sample-probe",
        "assets": {
            "member_dataset": args.member_dir.as_posix(),
            "nonmember_dataset": args.nonmember_dir.as_posix(),
            "model_dir": args.model_dir.as_posix(),
            "lora_dir": args.lora_dir.as_posix(),
            "clip_dir": args.clip_dir.as_posix(),
            "blip_dir": args.blip_dir.as_posix(),
        },
        "parameters": {
            "subset_limit": args.subset_limit,
            "num_returns": args.num_returns,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "resolution": args.resolution,
            "seed": args.seed,
            "caption_source": "metadata-with-blip-fallback",
        },
    }
    (args.run_root / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    start = time.time()
    runner = ProbeRunner(args)
    records = []
    for split_name, split_rows, split_dir, split_label, seed_offset in [
        ("member", member_rows, args.member_dir, 1, 1000),
        ("nonmember", nonmember_rows, args.nonmember_dir, 0, 2000),
    ]:
        for index, row in enumerate(split_rows, start=1):
            record = runner.feature_row(
                split_dir=split_dir,
                row=row,
                base_seed=args.seed + seed_offset + index * 17,
                output_dir=generation_dir,
                sample_index=len(records),
            )
            record["split"] = split_name
            record["label"] = split_label
            record["index"] = index
            records.append(record)
            print(
                f"[{split_name}] {index}/{len(split_rows)} {row['file_name']} "
                f"mean_cos={record['mean_cos']:.5f} max_cos={record['max_cos']:.5f} mean_ssim={record['mean_ssim']:.5f}"
            )

    features = np.asarray(
        [
            [
                row["mean_cos"],
                row["max_cos"],
                row["std_cos"],
                row["gap_cos"],
                row["mean_ssim"],
                row["max_ssim"],
            ]
            for row in records
        ],
        dtype=float,
    )
    labels = np.asarray([row["label"] for row in records], dtype=int)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)
    splits = min(4, int(labels.sum()), int((labels == 0).sum()))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=args.seed)
    probs = cross_val_predict(model, features, labels, cv=cv, method="predict_proba")[:, 1]
    metrics = search_threshold(labels, probs)
    model.fit(features, labels)
    elapsed_hours = (time.time() - start) / 3600.0

    summary = {
        "run_id": args.run_root.name,
        "track": "black-box",
        "method": "semantic-auxiliary-classifier",
        "dataset": "CelebA",
        "model": "StableDiffusion-v1.5 + Recon target LoRA",
        "mode": "small-sample-probe",
        "status": "candidate" if metrics["auc"] >= 0.6 else "negative but useful",
        "metrics": {
            "auc": round(float(metrics["auc"]), 6),
            "asr": round(float(metrics["asr"]), 6),
            "tpr_at_1pct_fpr": round(float(metrics["tpr_at_1pct_fpr"]), 6),
        },
        "analysis": {
            "threshold": round(float(metrics["threshold"]), 6),
            "best_tpr": round(float(metrics["best_tpr"]), 6),
            "best_fpr": round(float(metrics["best_fpr"]), 6),
            "member_mean_cos": round(float(np.mean([row["mean_cos"] for row in records if row["label"] == 1])), 6),
            "nonmember_mean_cos": round(float(np.mean([row["mean_cos"] for row in records if row["label"] == 0])), 6),
            "member_max_cos": round(float(np.mean([row["max_cos"] for row in records if row["label"] == 1])), 6),
            "nonmember_max_cos": round(float(np.mean([row["max_cos"] for row in records if row["label"] == 0])), 6),
            "member_rows": int(labels.sum()),
            "nonmember_rows": int((labels == 0).sum()),
            "feature_names": ["mean_cos", "max_cos", "std_cos", "gap_cos", "mean_ssim", "max_ssim"],
            "cv_splits": splits,
        },
        "cost": {
            "gpu_hours": round(float(elapsed_hours), 4),
        },
        "notes": (
            "Small-sample black-box probe using returned-image semantic features and an offline logistic auxiliary classifier. "
            "Generation uses only image-derived prompts and final returned images."
        ),
        "artifacts": {
            "records_json": (output_dir / "records.json").as_posix(),
            "generated_dir": generation_dir.as_posix(),
        },
    }

    (output_dir / "records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.run_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
