from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel

from diffaudit.attacks.prompt_response_consistency import (
    collect_generated_paths,
    cosine_similarity,
    summarize_prompt_response,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit prompt-response consistency on existing semantic-aux generated outputs.")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


class PromptResponseRunner:
    def __init__(self, clip_dir: Path, device: str) -> None:
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(clip_dir, local_files_only=True)
        self.model = CLIPModel.from_pretrained(clip_dir, torch_dtype=self.dtype, local_files_only=True).to(self.device)
        self.model.eval()

    def text_embedding(self, prompt: str) -> np.ndarray:
        inputs = self.processor(text=[prompt], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            text_outputs = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            features = self.model.text_projection(text_outputs.pooler_output)
        features = features / features.norm(dim=-1, keepdim=True)
        return features[0].detach().float().cpu().numpy()

    def image_embedding(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            vision_outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
            features = self.model.visual_projection(vision_outputs.pooler_output)
        features = features / features.norm(dim=-1, keepdim=True)
        return features[0].detach().float().cpu().numpy()


def main() -> None:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    records = json.loads(args.records.read_text(encoding="utf-8"))
    runner = PromptResponseRunner(args.clip_dir, args.device)

    prompt_response_records = []
    consistency_scores = []
    baseline_scores = []
    labels = []
    for sample_index, row in enumerate(records):
        prompt = str(row["prompt"])
        generated_paths = collect_generated_paths(args.generated_dir, sample_index, str(row["file_name"]))
        if not generated_paths:
            raise FileNotFoundError(f"Missing generated images for sample index {sample_index}: {row['file_name']}")
        text_embed = runner.text_embedding(prompt)
        image_embeds = [runner.image_embedding(Image.open(path).convert("RGB")) for path in generated_paths]
        prompt_cosines = np.asarray([cosine_similarity(text_embed, embed) for embed in image_embeds], dtype=float)
        mean_prompt_cos = float(prompt_cosines.mean())
        max_prompt_cos = float(prompt_cosines.max())
        prompt_response_records.append(
            {
                "sample_index": sample_index,
                "file_name": row["file_name"],
                "label": int(row["label"]),
                "prompt": prompt,
                "mean_prompt_cos": mean_prompt_cos,
                "max_prompt_cos": max_prompt_cos,
                "generated_paths": [path.as_posix() for path in generated_paths],
            }
        )
        consistency_scores.append(mean_prompt_cos)
        baseline_scores.append(float(row["mean_cos"]))
        labels.append(int(row["label"]))

    label_array = np.asarray(labels, dtype=int)
    consistency_array = np.asarray(consistency_scores, dtype=float)
    baseline_array = np.asarray(baseline_scores, dtype=float)
    summary = {
        "task_scope": "BB-1.4 prompt-response consistency probe",
        "track": "black-box",
        "method_family": "prompt-response-consistency",
        "device": args.device,
        "records_source": args.records.as_posix(),
        "generated_dir": args.generated_dir.as_posix(),
        "metrics": summarize_prompt_response(label_array, consistency_array, baseline_array),
        "means": {
            "member_mean_prompt_cos": round(float(consistency_array[label_array == 1].mean()), 6),
            "nonmember_mean_prompt_cos": round(float(consistency_array[label_array == 0].mean()), 6),
            "member_mean_cos": round(float(baseline_array[label_array == 1].mean()), 6),
            "nonmember_mean_cos": round(float(baseline_array[label_array == 0].mean()), 6),
        },
        "notes": "Existing generated outputs reused; no new image generation performed.",
    }
    (args.run_root / "records.json").write_text(json.dumps(prompt_response_records, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.run_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
