"""Evaluate SMP-LoRA privacy protection effect using GSA attack.

This script evaluates whether SMP-LoRA reduces membership inference attack
accuracy compared to the unprotected model.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

research_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(research_root / "src"))

from diffaudit.defenses.lora_ddpm import LoRALinear, inject_lora_into_unet
from diffusers import DDPMScheduler, UNet2DModel


def find_lora_layers(module: nn.Module, prefix: str = "") -> dict[str, LoRALinear]:
    """Find all LoRA layers in the model."""
    lora_layers = {}

    try:
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, LoRALinear):
                lora_layers[full_name] = child
            elif isinstance(child, nn.ModuleList):
                for i, item in enumerate(child):
                    item_name = f"{full_name}.{i}"
                    if isinstance(item, LoRALinear):
                        lora_layers[item_name] = item
                    else:
                        lora_layers.update(find_lora_layers(item, item_name))
            else:
                lora_layers.update(find_lora_layers(child, full_name))
    except TypeError:
        pass

    return lora_layers


class SimpleGSA:
    """Simplified Gradient-based Side-channel Attack for evaluation."""

    def __init__(self, model: nn.Module, noise_scheduler: DDPMScheduler):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.device = next(model.parameters()).device

    def compute_gradients(self, images: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute gradients for given images."""
        images = images.to(self.device).requires_grad_(True)
        labels = labels.to(self.device)

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (images.shape[0],), device=self.device
        )

        noise = torch.randn_like(images)
        noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)

        self.model.zero_grad()
        output = self.model(noisy_images, timestep=timesteps).sample

        loss = torch.nn.functional.mse_loss(output, noise, reduction="none")
        loss = loss.mean()
        loss.backward()

        grads = {}
        if images.grad is not None:
            grads["input_grad"] = images.grad.detach()

        return grads

    def extract_features(self, gradients: dict[str, torch.Tensor]) -> np.ndarray:
        """Extract gradient-based features for attack."""
        features = []

        if "input_grad" in gradients:
            g = gradients["input_grad"]
            features.extend([
                g.abs().mean().item(),
                g.abs().std().item(),
                g.abs().max().item(),
                g.abs().min().item(),
            ])

            flat_grad = g.view(g.size(0), -1)
            features.extend([
                torch.norm(flat_grad, p=1, dim=1).mean().item(),
                torch.norm(flat_grad, p=2, dim=1).mean().item(),
            ])

        return np.array(features)

    def evaluate(
        self,
        member_images: list[Path],
        nonmember_images: list[Path],
        num_samples: int = 200,
        evaluation_seed: int = 42,
        score_artifact_path: Path | None = None,
    ) -> dict[str, Any]:
        """Evaluate attack accuracy on member vs non-member samples."""
        from torch.utils.data import DataLoader, Dataset

        class ImageDataset(Dataset):
            def __init__(self, paths):
                self.paths = paths

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                img = Image.open(self.paths[idx]).convert("RGB")
                img = img.resize((32, 32))
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = (arr - 0.5) / 0.5
                return torch.from_numpy(arr).permute(2, 0, 1)

        member_dataset = ImageDataset(member_images[:num_samples])
        nonmember_dataset = ImageDataset(nonmember_images[:num_samples])

        member_loader = DataLoader(member_dataset, batch_size=8, shuffle=False)
        nonmember_loader = DataLoader(nonmember_dataset, batch_size=8, shuffle=False)

        member_features = []
        nonmember_features = []

        for batch in member_loader:
            grads = self.compute_gradients(batch, torch.zeros(len(batch), device=self.device))
            feats = self.extract_features(grads)
            member_features.append(feats)

        for batch in nonmember_loader:
            grads = self.compute_gradients(batch, torch.zeros(len(batch), device=self.device))
            feats = self.extract_features(grads)
            nonmember_features.append(feats)

        member_features = np.array(member_features)
        nonmember_features = np.array(nonmember_features)

        X = np.vstack([member_features, nonmember_features])
        y = np.array([1] * len(member_features) + [0] * len(nonmember_features))

        rng = np.random.default_rng(evaluation_seed)
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]

        split = int(0.7 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            fpr, tpr, _ = roc_curve(y_test, y_prob)

            def _tpr_at(target_fpr: float) -> float:
                return float(tpr[np.argmin(np.abs(fpr - target_fpr))])

            result = {
                "accuracy": float(accuracy),
                "asr": float(accuracy),
                "auc": float(auc),
                "tpr_at_1pct_fpr": _tpr_at(0.01),
                "tpr_at_0_1pct_fpr": _tpr_at(0.001),
                "num_member": len(member_features),
                "num_nonmember": len(nonmember_features),
                "evaluation_seed": int(evaluation_seed),
                "num_train": int(len(X_train)),
                "num_test": int(len(X_test)),
            }

            if score_artifact_path is not None:
                score_artifact_path.parent.mkdir(parents=True, exist_ok=True)
                score_payload = {
                    "evaluation_seed": int(evaluation_seed),
                    "y_test": [int(v) for v in y_test.tolist()],
                    "y_pred": [int(v) for v in y_pred.tolist()],
                    "y_prob": [float(v) for v in y_prob.tolist()],
                    "fpr": [float(v) for v in fpr.tolist()],
                    "tpr": [float(v) for v in tpr.tolist()],
                }
                score_artifact_path.write_text(
                    json.dumps(score_payload, indent=2),
                    encoding="utf-8",
                )

            return result
        except ImportError:
            diff = member_features.mean(axis=0) - nonmember_features.mean(axis=0)
            score = np.linalg.norm(diff)

            return {
                "accuracy": float(score > 0.5),
                "auc": float(score),
                "num_member": len(member_features),
                "num_nonmember": len(nonmember_features),
                "note": "sklearn not available, using simplified metric",
            }


def create_ddpm_model() -> UNet2DModel:
    """Create a DDPM model matching training architecture."""
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


def load_base_model(model_path: Path, device: torch.device) -> UNet2DModel:
    """Load base DDPM model from local checkpoint."""
    model = create_ddpm_model().to(device)
    safetensors_path = model_path / "model.safetensors"

    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_path), device=str(device))
            model.load_state_dict(state_dict)
            print(f"Loaded model from {safetensors_path}")
        except Exception as e:
            print(f"Failed to load safetensors: {e}, using random init")
    else:
        print(f"No model.safetensors found at {model_path}, using random init")

    return model


def load_lora_weights_direct(model: nn.Module, lora_state: dict[str, torch.Tensor], device: torch.device) -> nn.Module:
    """Directly load LoRA weights by matching keys to LoRA layers."""
    lora_layers = find_lora_layers(model)

    for lora_layer in lora_layers.values():
        lora_layer.to(device)

    for key, tensor in lora_state.items():
        parts = key.split(".")
        if len(parts) < 2:
            continue

        lora_name = ".".join(parts[:-1])
        param_name = parts[-1]

        if lora_name in lora_layers:
            layer = lora_layers[lora_name]
            if param_name == "lora_A":
                layer.lora_A.data.copy_(tensor.to(device))
            elif param_name == "lora_B":
                layer.lora_B.data.copy_(tensor.to(device))

    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate SMP-LoRA privacy protection")
    parser.add_argument("--lora_checkpoint", type=Path, required=True, help="Path to LoRA weights")
    parser.add_argument("--base_model", type=Path, default=None, help="Path to base model (for pretrained models)")
    parser.add_argument("--member_dir", type=Path, required=True, help="Member images directory")
    parser.add_argument("--nonmember_dir", type=Path, required=True, help="Non-member images directory")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument(
        "--details_output",
        type=Path,
        default=None,
        help="Optional path for persisted score/probability artifacts",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to evaluate")
    parser.add_argument("--rank", type=int, default=None, help="LoRA rank (auto-detected from config if not specified)")
    parser.add_argument(
        "--evaluation_seed",
        type=int,
        default=42,
        help="Seed used for evaluation-set permutation before train/test split",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    member_images = sorted(args.member_dir.glob("*.png"))
    nonmember_images = sorted(args.nonmember_dir.glob("*.png"))

    print(f"Found {len(member_images)} member images, {len(nonmember_images)} non-member images")

    lora_rank = args.rank
    if lora_rank is None:
        config_path = args.lora_checkpoint.parent.parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            lora_rank = config.get("rank", 4)
            print(f"Auto-detected rank={lora_rank} from config")
        else:
            lora_state = torch.load(args.lora_checkpoint, map_location="cpu", weights_only=True)
            for key, tensor in lora_state.items():
                if "lora_A" in key:
                    lora_rank = tensor.shape[-1]
                    print(f"Auto-detected rank={lora_rank} from weights")
                    break
            if lora_rank is None:
                lora_rank = 4
                print(f"Using default rank={lora_rank}")

    if args.base_model:
        print(f"Loading base model from: {args.base_model}")
        model = load_base_model(args.base_model, device)
    else:
        print("Using random initialized base model")
        model = create_ddpm_model().to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon",
    )

    print(f"Injecting LoRA layers with rank={lora_rank}...")
    injected = inject_lora_into_unet(model, rank=lora_rank, alpha=1.0)
    print(f"Injected {len(injected)} LoRA layers")

    print("Loading LoRA weights...")
    lora_state = torch.load(args.lora_checkpoint, map_location=device, weights_only=True)
    model = load_lora_weights_direct(model, lora_state, device)

    for lora_layer in find_lora_layers(model).values():
        lora_layer.lora_A.requires_grad_(True)
        lora_layer.lora_B.requires_grad_(True)

    print("Evaluating protected model...")
    gsa = SimpleGSA(model, noise_scheduler)

    details_output = args.details_output
    if details_output is None and args.output is not None:
        details_output = args.output.with_name(f"{args.output.stem}_details.json")

    result = gsa.evaluate(
        member_images,
        nonmember_images,
        num_samples=args.num_samples,
        evaluation_seed=args.evaluation_seed,
        score_artifact_path=details_output,
    )

    print("\n=== Evaluation Results (LoRA-Protected) ===")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"ASR: {result['asr']:.4f}")
    print(f"AUC: {result['auc']:.4f}")
    print(f"TPR@1%FPR: {result['tpr_at_1pct_fpr']:.4f}")
    print(f"TPR@0.1%FPR: {result['tpr_at_0_1pct_fpr']:.4f}")
    print(f"Member samples: {result['num_member']}")
    print(f"Non-member samples: {result['num_nonmember']}")
    print(f"Evaluation seed: {result['evaluation_seed']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")
    if details_output is not None:
        print(f"Detailed score artifacts saved to {details_output}")

    return result


if __name__ == "__main__":
    main()
