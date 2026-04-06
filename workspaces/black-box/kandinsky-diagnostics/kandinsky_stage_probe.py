from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from datasets import Dataset
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
from diffusers.models import UNet2DConditionModel
from safetensors.torch import load_file as load_safetensors
from transformers import CLIPVisionModelWithProjection


def load_weights(path: str):
    if str(path).endswith(".safetensors"):
        return load_safetensors(path)
    return torch.load(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder-dir", required=True)
    parser.add_argument("--prior-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--inference-steps", type=int, default=30)
    return parser.parse_args()


def stamp(label: str, start: float) -> None:
    print(f"{label}: {time.perf_counter() - start:.2f}s", flush=True)


def main() -> None:
    args = parse_args()
    device = f"cuda:{args.gpu}"
    torch_dtype = torch.float16

    overall = time.perf_counter()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior",
        subfolder="image_encoder",
    ).to(torch_dtype).to(device)
    stamp("loaded image_encoder", overall)

    unet = UNet2DConditionModel.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder",
        subfolder="unet",
    ).to(torch_dtype).to(device)
    stamp("loaded unet", overall)

    prior = KandinskyV22PriorPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior",
        image_encoder=image_encoder,
        torch_dtype=torch_dtype,
    ).to(device)
    stamp("loaded prior pipeline", overall)

    decoder = KandinskyV22Pipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder",
        unet=unet,
        torch_dtype=torch_dtype,
    ).to(device)
    stamp("loaded decoder pipeline", overall)

    decoder.unet.load_attn_procs(load_weights(args.decoder_dir))
    stamp("loaded decoder lora", overall)
    prior.prior.load_attn_procs(load_weights(args.prior_dir))
    stamp("loaded prior lora", overall)

    dataset = Dataset.from_dict(torch.load(args.dataset_dir))
    prompt = dataset["text"][0][:77]
    print(f"prompt={prompt!r}", flush=True)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    step = time.perf_counter()
    img_emb = prior(
        prompt=prompt,
        num_inference_steps=args.inference_steps,
        num_images_per_prompt=1,
    )
    stamp("prior forward", step)

    negative_prior_prompt = (
        "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, "
        "duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly "
        "drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, "
        "bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed "
        "limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many "
        "fingers, long neck, username, watermark, signature"
    )
    step = time.perf_counter()
    negative_emb = prior(
        prompt=negative_prior_prompt,
        num_inference_steps=min(args.inference_steps, 25),
        num_images_per_prompt=1,
    )
    stamp("negative prior forward", step)

    step = time.perf_counter()
    images = decoder(
        image_embeds=img_emb.image_embeds,
        negative_image_embeds=negative_emb.image_embeds,
        num_inference_steps=args.inference_steps,
        height=512,
        width=512,
        guidance_scale=7.5,
    )
    stamp("decoder forward", step)

    step = time.perf_counter()
    save_path = save_dir / "image_01_01.jpg"
    images.images[0].save(save_path)
    stamp("image save", step)
    stamp("overall", overall)
    print(f"saved={save_path}", flush=True)


if __name__ == "__main__":
    main()
