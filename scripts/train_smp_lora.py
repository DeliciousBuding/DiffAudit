"""SMP-LoRA training script for DDPM on CIFAR-10.

Usage:
    # 使用本地模型
    python scripts/train_smp_lora.py \
        --local_model workspaces/white-box/assets/gsa-gpu-128/checkpoints/target/checkpoint-64 \
        --member_dir workspaces/white-box/assets/gsa-cifar10-1k-3shadow/datasets/target-member \
        --nonmember_dir workspaces/white-box/assets/gsa-cifar10-1k-3shadow/datasets/target-nonmember \
        --output_dir outputs/smp-lora-cifar10 \
        --rank 4 \
        --lambda_coeff 0.5 \
        --num_epochs 10 \
        --batch_size 8 \
        --device cuda

    # 使用 HuggingFace 模型（需要网络）
    python scripts/train_smp_lora.py \
        --pretrained_model hf://google/ddpm-cifar10-32 \
        --member_dir workspaces/white-box/assets/gsa-cifar10-1k-3shadow/datasets/target-member \
        --nonmember_dir workspaces/white-box/assets/gsa-cifar10-1k-3shadow/datasets/target-nonmember \
        --output_dir outputs/smp-lora-cifar10 \
        --rank 4 \
        --lambda_coeff 0.5 \
        --num_epochs 10 \
        --batch_size 8 \
        --device cuda

    # 使用随机初始化模型（无预训练）
    python scripts/train_smp_lora.py \
        --random_init \
        --member_dir workspaces/white-box/assets/gsa-cifar10-1k-3shadow/datasets/target-member \
        --nonmember_dir workspaces/white-box/assets/gsa-cifar10-1k-3shadow/datasets/target-nonmember \
        --output_dir outputs/smp-lora-cifar10 \
        --rank 4 \
        --lambda_coeff 0.5 \
        --num_epochs 10 \
        --batch_size 8 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import random
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as tv_transforms
from diffusers import DDPMScheduler, UNet2DModel
from torch.utils.data import DataLoader, Dataset

from diffaudit.defenses.lora_ddpm import (
    get_lora_state_dict,
    inject_lora_into_unet,
    lora_injection_summary,
)
from diffaudit.defenses.smp_lora import ProxyAttackModel, SMPLoRATrainer


def recommended_num_workers(cpu_count: int | None = None, cap: int = 8) -> int:
    """Pick a conservative worker count for a single-GPU training job."""
    detected = os.cpu_count() if cpu_count is None else cpu_count
    if detected is None or detected <= 1:
        return 0
    return min(cap, max(0, detected // 2))


def build_dataloader_kwargs(
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> dict[str, int | bool]:
    kwargs: dict[str, int | bool] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return kwargs


def build_runtime_config(
    device: str | torch.device,
    throughput_mode: bool,
    amp_dtype: str,
    allow_tf32: bool,
    cudnn_benchmark: bool,
    non_blocking_transfers: bool,
) -> dict[str, object]:
    resolved_device = torch.device(device)
    is_cuda = resolved_device.type == "cuda"
    config: dict[str, object] = {
        "device_type": resolved_device.type,
        "throughput_mode": bool(throughput_mode and is_cuda),
        "autocast_enabled": False,
        "autocast_dtype": None,
        "allow_tf32": False,
        "cudnn_benchmark": False,
        "non_blocking_transfers": False,
        "matmul_precision": "highest",
    }
    if not config["throughput_mode"]:
        return config

    amp_map = {
        "none": None,
        "bf16": torch.bfloat16,
    }
    resolved_amp_dtype = amp_map[amp_dtype]
    config["autocast_enabled"] = resolved_amp_dtype is not None
    config["autocast_dtype"] = resolved_amp_dtype
    config["allow_tf32"] = bool(allow_tf32)
    config["cudnn_benchmark"] = bool(cudnn_benchmark)
    config["non_blocking_transfers"] = bool(non_blocking_transfers)
    config["matmul_precision"] = "high"
    return config


def apply_runtime_config(runtime_config: dict[str, object]) -> None:
    if runtime_config["device_type"] != "cuda":
        return
    torch.set_float32_matmul_precision(str(runtime_config["matmul_precision"]))
    torch.backends.cuda.matmul.allow_tf32 = bool(runtime_config["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(runtime_config["allow_tf32"])
    torch.backends.cudnn.benchmark = bool(runtime_config["cudnn_benchmark"])


def move_tensor_to_device(
    tensor: torch.Tensor,
    device: torch.device,
    non_blocking: bool,
) -> torch.Tensor:
    return tensor.to(device, non_blocking=non_blocking)


def serialize_runtime_config(runtime_config: dict[str, object]) -> dict[str, object]:
    serialized = dict(runtime_config)
    if isinstance(serialized.get("autocast_dtype"), torch.dtype):
        serialized["autocast_dtype"] = str(serialized["autocast_dtype"]).replace(
            "torch.", ""
        )
    return serialized


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_seed(seed: int | None) -> bool:
    if seed is None:
        return False
    set_global_seed(seed)
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train SMP-LoRA on CIFAR-10 with DDPM")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="",
        help="Pretrained model identifier or path",
    )
    parser.add_argument(
        "--local_model",
        type=str,
        default="",
        help="Local directory containing model.safetensors",
    )
    parser.add_argument(
        "--random_init",
        action="store_true",
        default=False,
        help="Use randomly initialized model instead of pretrained",
    )
    parser.add_argument(
        "--member_dir",
        type=str,
        required=True,
        help="Directory containing member PNG images",
    )
    parser.add_argument(
        "--nonmember_dir",
        type=str,
        required=True,
        help="Directory containing non-member PNG images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/smp-lora-cifar10",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lambda_coeff", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=1e-4)
    parser.add_argument("--lora_lr", type=float, default=1e-4)
    parser.add_argument("--proxy_lr", type=float, default=1e-3)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "adamw", "sgd"],
        help="Optimizer applied to both LoRA and proxy updates",
    )
    parser.add_argument(
        "--sgd_momentum",
        type=float,
        default=0.9,
        help="Momentum used when --optimizer=sgd",
    )
    parser.add_argument("--proxy_hidden_dim", type=int, default=256)
    parser.add_argument("--proxy_steps", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=recommended_num_workers())
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--method", type=str, default="smp", choices=["smp", "mp"])
    parser.add_argument("--ddpm_num_train_timesteps", type=int, default=1000)
    parser.add_argument(
        "--throughput_mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable throughput-oriented CUDA defaults for future launches",
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="bf16",
        choices=["none", "bf16"],
        help="Autocast dtype used when throughput mode is enabled on CUDA",
    )
    parser.add_argument(
        "--allow_tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow TF32 matmul/cudnn kernels under throughput mode",
    )
    parser.add_argument(
        "--cudnn_benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable cudnn benchmark under throughput mode",
    )
    parser.add_argument(
        "--non_blocking_transfers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use non-blocking host-to-device copies when possible",
    )
    parser.add_argument("--save_every", type=int, default=500)
    return parser.parse_args()


class ImageFileDataset(Dataset):
    """Dataset for loading images from a flat directory."""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = sorted([p for p in self.root_dir.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        if not self.image_paths:
            raise FileNotFoundError(f"No image files found in {root_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0, str(img_path)


def load_cifar10_split(data_dir: str, batch_size: int, num_workers: int, shuffle: bool = True):
    transform = tv_transforms.Compose([
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = ImageFileDataset(root_dir=data_dir, transform=transform)
    loader = DataLoader(dataset, **build_dataloader_kwargs(batch_size, shuffle, num_workers))
    return loader


def create_ddpm_model():
    return UNet2DModel(
        sample_size=32,
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

def load_ddpm_model(args):
    if args.random_init:
        print("Using randomly initialized DDPM model")
        return create_ddpm_model()
    elif args.local_model:
        print(f"Loading local model from: {args.local_model}")
        model = create_ddpm_model()
        model_path = Path(args.local_model) / "model.safetensors"
        if not model_path.exists():
            raise FileNotFoundError(f"model.safetensors not found at {model_path}")
        try:
            from safetensors import safe_open
            with safe_open(model_path, framework="pt") as f:
                state_dict = {k: f.get_tensor(k) for k in f.keys()}
            model.load_state_dict(state_dict)
        except ImportError:
            print("safetensors not installed, trying torch.load...")
            try:
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
            except Exception as e:
                print(f"Failed to load model: {e}")
                print("Using randomly initialized model instead")
                return create_ddpm_model()
    elif args.pretrained_model:
        print(f"Loading pretrained model: {args.pretrained_model}")
        model = UNet2DModel.from_pretrained(args.pretrained_model)
    else:
        raise ValueError("Either --local_model, --pretrained_model, or --random_init must be specified")
    return model


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_applied = apply_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if seed_applied:
        print(f"Seed: {args.seed}")
    else:
        print("Seed: None (historical unseeded mode)")

    runtime_config = build_runtime_config(
        device=device,
        throughput_mode=args.throughput_mode,
        amp_dtype=args.amp_dtype,
        allow_tf32=args.allow_tf32,
        cudnn_benchmark=args.cudnn_benchmark,
        non_blocking_transfers=args.non_blocking_transfers,
    )
    apply_runtime_config(runtime_config)
    print("Runtime config:")
    print(f"  throughput_mode={runtime_config['throughput_mode']}")
    print(f"  autocast_enabled={runtime_config['autocast_enabled']}")
    print(f"  autocast_dtype={runtime_config['autocast_dtype']}")
    print(f"  allow_tf32={runtime_config['allow_tf32']}")
    print(f"  cudnn_benchmark={runtime_config['cudnn_benchmark']}")
    print(f"  non_blocking_transfers={runtime_config['non_blocking_transfers']}")

    model = load_ddpm_model(args)
    model = model.to(device)

    print(f"Loading member data from: {args.member_dir}")
    member_loader = load_cifar10_split(
        args.member_dir, args.batch_size, args.num_workers, shuffle=True
    )
    print(f"Loading non-member data from: {args.nonmember_dir}")
    nonmember_loader = load_cifar10_split(
        args.nonmember_dir, args.batch_size, args.num_workers, shuffle=True
    )

    member_iter = iter(member_loader)
    nonmember_iter = iter(nonmember_loader)

    print(f"Initializing SMP-LoRA trainer:")
    print(f"  rank={args.rank}, alpha={args.alpha}, lambda={args.lambda_coeff}")
    print(f"  method={args.method}, proxy_steps={args.proxy_steps}")
    print(
        f"  optimizer={args.optimizer}, lora_lr={args.lora_lr}, proxy_lr={args.proxy_lr}"
    )
    if args.optimizer == "sgd":
        print(f"  sgd_momentum={args.sgd_momentum}")

    trainer = SMPLoRATrainer(
        model=model,
        rank=args.rank,
        alpha=args.alpha,
        lambda_coeff=args.lambda_coeff,
        delta=args.delta,
        lora_lr=args.lora_lr,
        proxy_lr=args.proxy_lr,
        optimizer=args.optimizer,
        sgd_momentum=args.sgd_momentum,
        proxy_hidden_dim=args.proxy_hidden_dim,
        proxy_steps=args.proxy_steps,
        method=args.method,
        ddpm_num_train_timesteps=args.ddpm_num_train_timesteps,
        device=str(device),
    )

    print(f"Model injection summary:")
    summary = lora_injection_summary(model)
    print(f"  LoRA layers: {summary['num_lora_layers']}")
    print(f"  LoRA params: {summary['total_lora_params']:,}")
    print(f"  Compression ratio: {summary['overall_compression_ratio']:.1f}x")

    config = vars(args)
    config["device"] = str(device)
    config["lora_summary"] = summary
    config["runtime_config"] = serialize_runtime_config(runtime_config)
    config["training_started_at"] = datetime.now(timezone.utc).isoformat()

    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"Saved config to {config_path}")

    global_step = 0
    autocast_context = nullcontext
    if runtime_config["autocast_enabled"]:
        autocast_context = lambda: torch.autocast(
            device_type=str(runtime_config["device_type"]),
            dtype=runtime_config["autocast_dtype"],
        )
    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in range(args.num_epochs):
        for batch_idx in range(len(member_loader)):
            try:
                member_batch, _, _ = next(member_iter)
            except StopIteration:
                member_iter = iter(member_loader)
                member_batch, _, _ = next(member_iter)

            try:
                nonmember_batch, _, _ = next(nonmember_iter)
            except StopIteration:
                nonmember_iter = iter(nonmember_loader)
                nonmember_batch, _, _ = next(nonmember_iter)

            member_batch = move_tensor_to_device(
                member_batch,
                device=device,
                non_blocking=bool(runtime_config["non_blocking_transfers"]),
            )
            nonmember_batch = move_tensor_to_device(
                nonmember_batch,
                device=device,
                non_blocking=bool(runtime_config["non_blocking_transfers"]),
            )
            timesteps = torch.randint(
                0, args.ddpm_num_train_timesteps, (member_batch.shape[0],), device=device
            )

            with autocast_context():
                record = trainer.train_step(
                    member_batch=member_batch,
                    nonmember_batch=nonmember_batch,
                    timesteps=timesteps,
                    step=global_step,
                )

            if global_step % 10 == 0:
                print(
                    f"Epoch {epoch}/{args.num_epochs} | "
                    f"Step {global_step} | "
                    f"loss={record['adaptation_loss']:.6f} | "
                    f"mi_gain={record['mi_gain']:.4f} | "
                    f"objective={record['objective']:.6f}"
                )

            if (global_step + 1) % args.save_every == 0:
                checkpoint_dir = output_dir / f"step_{global_step + 1}"
                trainer.save_checkpoint(checkpoint_dir, include_training_log=False)
                print(f"Saved checkpoint to {checkpoint_dir}")

            global_step += 1

    final_dir = output_dir / "final"
    trainer.save_checkpoint(final_dir)
    print(f"Training complete. Final checkpoint saved to {final_dir}")
    print(f"Total steps: {global_step}")
    print(f"Total log entries: {len(trainer.log)}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
