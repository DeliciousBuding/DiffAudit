"""Phase 2: Multi-model validation and baseline comparison for SMP-LoRA.

This script runs:
1. SMP-LoRA training on all available shadow/target models
2. No-defense baseline GSA evaluation
3. Plain LoRA (no min-max) baseline GSA evaluation
4. Long training run with best config
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

research_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(research_root / "src"))

CHECKPOINTS_BASE = Path("D:/Code/DiffAudit/Research/workspaces/white-box/assets/gsa-gpu-128/checkpoints")
MEMBER_DIR = "D:/Code/DiffAudit/Research/workspaces/white-box/assets/gsa-cifar10-1k-3shadow/datasets/target-member"
NONMEMBER_DIR = "D:/Code/DiffAudit/Research/workspaces/white-box/assets/gsa-cifar10-1k-3shadow/datasets/target-nonmember"
OUTPUT_BASE = Path("D:/Code/DiffAudit/Research/outputs/smp-lora-phase2")

MODELS = {
    "target-64": CHECKPOINTS_BASE / "target" / "checkpoint-64",
    "target-33": CHECKPOINTS_BASE / "target" / "checkpoint-33",
    "shadow-64": CHECKPOINTS_BASE / "shadow" / "checkpoint-64",
    "shadow-33": CHECKPOINTS_BASE / "shadow" / "checkpoint-33",
}


def run_command(cmd: list[str], desc: str, timeout: int = 7200) -> tuple[bool, str]:
    """Run a command and return (success, output)."""
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {desc}")
    print(f"{'='*60}")

    start = time.time()
    try:
        result = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
        elapsed = time.time() - start
        print(f"Done in {elapsed:.1f}s, exit={result.returncode}")
        if result.returncode != 0:
            print(f"STDERR: {result.stderr[-500:]}")
        return result.returncode == 0, result.stdout[-500:] if result.stdout else ""
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {timeout}s")
        return False, "timeout"
    except Exception as e:
        print(f"ERROR: {e}")
        return False, str(e)


def train_smp_lora(model_name: str, model_path: Path, lambda_coeff: float, rank: int, num_epochs: int) -> Path | None:
    """Train SMP-LoRA on a specific model."""
    output_dir = OUTPUT_BASE / f"smp_{model_name}_l{lambda_coeff}_r{rank}_e{num_epochs}"
    lora_checkpoint = output_dir / "final" / "lora_weights.pt"

    if lora_checkpoint.exists():
        print(f"Skip training (exists): {lora_checkpoint}")
        return lora_checkpoint

    cmd = [
        "conda", "run", "-n", "diffaudit-research",
        "python", "D:/Code/DiffAudit/Research/scripts/train_smp_lora.py",
        "--local_model", str(model_path),
        "--member_dir", MEMBER_DIR,
        "--nonmember_dir", NONMEMBER_DIR,
        "--output_dir", str(output_dir),
        "--rank", str(rank),
        "--lambda_coeff", str(lambda_coeff),
        "--num_epochs", str(num_epochs),
        "--batch_size", "8",
        "--device", "cuda",
        "--save_every", "100",
    ]

    success, _ = run_command(cmd, f"Train SMP-LoRA: {model_name}", timeout=num_epochs * 600 + 600)
    return lora_checkpoint if success else None


def evaluate_smp_lora(name: str, lora_checkpoint: Path, base_model: Path) -> dict[str, Any]:
    """Evaluate SMP-LoRA defense effect."""
    eval_output = lora_checkpoint.parent.parent / "evaluation.json"

    cmd = [
        "conda", "run", "-n", "diffaudit-research",
        "python", "D:/Code/DiffAudit/Research/scripts/evaluate_smp_lora_defense.py",
        "--lora_checkpoint", str(lora_checkpoint),
        "--base_model", str(base_model),
        "--member_dir", MEMBER_DIR,
        "--nonmember_dir", NONMEMBER_DIR,
        "--device", "cuda",
        "--num_samples", "500",
        "--output", str(eval_output),
    ]

    success, _ = run_command(cmd, f"Eval: {name}", timeout=600)

    if success and eval_output.exists():
        with open(eval_output) as f:
            return json.load(f)
    return {"accuracy": None, "auc": None, "error": "eval failed"}


def evaluate_no_defense_baseline(model_name: str, model_path: Path) -> dict[str, Any]:
    """Evaluate GSA attack on model without any defense."""
    output_dir = OUTPUT_BASE / f"baseline_nodefense_{model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_output = output_dir / "evaluation.json"

    script = f"""
import sys
sys.path.insert(0, 'D:/Code/DiffAudit/Research/src')
import torch
from pathlib import Path
from diffusers import UNet2DModel, DDPMScheduler
from safetensors.torch import load_file
import numpy as np
from PIL import Image

device = torch.device('cuda')
model = UNet2DModel(
    sample_size=32, in_channels=3, out_channels=3, layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=('DownBlock2D','DownBlock2D','DownBlock2D','DownBlock2D','AttnDownBlock2D','DownBlock2D'),
    up_block_types=('UpBlock2D','AttnUpBlock2D','UpBlock2D','UpBlock2D','UpBlock2D','UpBlock2D'),
).to(device)

st = load_file(r'{model_path}/model.safetensors', device=str(device))
model.load_state_dict(st)

scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='linear', prediction_type='epsilon')

from scripts.evaluate_smp_lora_defense import SimpleGSA
gsa = SimpleGSA(model, scheduler)
member = sorted(Path(r'{MEMBER_DIR}').glob('*.png'))
nonmember = sorted(Path(r'{NONMEMBER_DIR}').glob('*.png'))
result = gsa.evaluate(member, nonmember, num_samples=500)

import json
with open(r'{eval_output}', 'w') as f:
    json.dump(result, f, indent=2)
print(f"No-defense baseline: Acc={{result['accuracy']:.4f}}, AUC={{result['auc']:.4f}}")
"""
    script_path = output_dir / "eval_baseline.py"
    with open(script_path, "w") as f:
        f.write(script)

    cmd = ["conda", "run", "-n", "diffaudit-research", "python", str(script_path)]
    success, _ = run_command(cmd, f"No-defense baseline: {model_name}", timeout=600)

    if success and eval_output.exists():
        with open(eval_output) as f:
            return json.load(f)
    return {"accuracy": None, "auc": None, "error": "baseline eval failed"}


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Multi-model + baseline")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--long_epochs", type=int, default=100)
    args = parser.parse_args()

    results = []
    start_time = datetime.now()

    best_lambda = 0.5
    best_rank = 4
    best_epochs = 20

    print(f"\n{'#'*60}")
    print(f"# Phase 2: Multi-model + Baseline Comparison")
    print(f"# Models: {list(MODELS.keys())}")
    print(f"# Start: {start_time}")
    print(f"{'#'*60}")

    # Step 1: SMP-LoRA on all models
    if not args.skip_training:
        print("\n## Step 1: SMP-LoRA on all models")
        for model_name, model_path in MODELS.items():
            if not model_path.exists():
                print(f"Skip {model_name}: path not found")
                continue

            ckpt = train_smp_lora(model_name, model_path, best_lambda, best_rank, best_epochs)
            if ckpt:
                if not args.skip_eval:
                    result = evaluate_smp_lora(f"smp_{model_name}", ckpt, model_path)
                    result["model"] = model_name
                    result["defense"] = "smp-lora"
                    results.append(result)

    # Step 2: No-defense baselines
    if not args.skip_baseline:
        print("\n## Step 2: No-defense baselines")
        for model_name, model_path in MODELS.items():
            if not model_path.exists():
                continue
            result = evaluate_no_defense_baseline(model_name, model_path)
            result["model"] = model_name
            result["defense"] = "none"
            results.append(result)

    # Step 3: Long training with best config
    if not args.skip_training:
        print("\n## Step 3: Long training run")
        target_path = MODELS["target-64"]
        if target_path.exists():
            ckpt = train_smp_lora("target-64-long", target_path, best_lambda, best_rank, args.long_epochs)
            if ckpt and not args.skip_eval:
                result = evaluate_smp_lora("smp_target-64-long", ckpt, target_path)
                result["model"] = "target-64"
                result["defense"] = f"smp-lora-ep{args.long_epochs}"
                results.append(result)

    # Save results
    results_file = OUTPUT_BASE / "phase2_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "results": results,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\n{'#'*60}")
    print(f"# Phase 2 Complete!")
    print(f"# Results: {results_file}")
    print(f"{'#'*60}")

    for r in results:
        if r.get("auc") is not None:
            print(f"  {r.get('defense','?'):20s} | {r.get('model','?'):12s} | AUC={r['auc']:.4f} | Acc={r['accuracy']:.4f}")

    return results


if __name__ == "__main__":
    main()
