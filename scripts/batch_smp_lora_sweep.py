"""Batch SMP-LoRA training and evaluation script for hyperparameter sweep.

This script runs a comprehensive sweep over lambda, rank, and epochs
to find the optimal SMP-LoRA configuration for privacy-utility tradeoff.
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

BASE_MODEL = str(research_root / "workspaces" / "white-box" / "assets" / "gsa-gpu-128" / "checkpoints" / "target" / "checkpoint-64")
MEMBER_DIR = str(research_root / "workspaces" / "white-box" / "assets" / "gsa-cifar10-1k-3shadow" / "datasets" / "target-member")
NONMEMBER_DIR = str(research_root / "workspaces" / "white-box" / "assets" / "gsa-cifar10-1k-3shadow" / "datasets" / "target-nonmember")
OUTPUT_BASE = str(research_root / "outputs" / "smp-lora-sweep")


def run_command(cmd: list[str], desc: str, timeout: int = 7200) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {desc}")
    print(f"Command: {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - start
        print(f"Completed in {elapsed:.1f}s, exit code: {result.returncode}")

        if result.returncode != 0:
            print(f"STDERR: {result.stderr[-1000:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def train_smp_lora(
    name: str,
    lambda_coeff: float,
    rank: int,
    num_epochs: int,
) -> Path:
    """Train SMP-LoRA with given parameters."""
    output_dir = Path(OUTPUT_BASE) / name
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_checkpoint = output_dir / "final" / "lora_weights.pt"

    if lora_checkpoint.exists():
        print(f"Checkpoint exists, skipping training: {lora_checkpoint}")
        return lora_checkpoint

    cmd = [
        "conda", "run", "-n", "diffaudit-research",
        "python", str(research_root / "scripts" / "train_smp_lora.py"),
        "--local_model", BASE_MODEL,
        "--member_dir", MEMBER_DIR,
        "--nonmember_dir", NONMEMBER_DIR,
        "--output_dir", str(output_dir),
        "--rank", str(rank),
        "--lambda_coeff", str(lambda_coeff),
        "--num_epochs", str(num_epochs),
        "--batch_size", "8",
        "--device", "cuda",
        "--save_every", "50",
    ]

    success = run_command(cmd, f"Training {name}", timeout=num_epochs * 600 + 600)
    return lora_checkpoint if success else None


def evaluate_smp_lora(
    name: str,
    lora_checkpoint: Path,
    lambda_coeff: float,
    rank: int,
    num_epochs: int,
) -> dict[str, Any]:
    """Evaluate SMP-LoRA with given parameters."""
    eval_output = lora_checkpoint.parent.parent / "evaluation_pretrained.json"

    cmd = [
        "conda", "run", "-n", "diffaudit-research",
        "python", str(research_root / "scripts" / "evaluate_smp_lora_defense.py"),
        "--lora_checkpoint", str(lora_checkpoint),
        "--base_model", BASE_MODEL,
        "--member_dir", MEMBER_DIR,
        "--nonmember_dir", NONMEMBER_DIR,
        "--device", "cuda",
        "--num_samples", "500",
        "--output", str(eval_output),
    ]

    success = run_command(cmd, f"Evaluating {name}", timeout=600)

    if success and eval_output.exists():
        with open(eval_output) as f:
            result = json.load(f)
        result["config"] = {
            "lambda": lambda_coeff,
            "rank": rank,
            "epochs": num_epochs,
        }
        return result

    return {
        "config": {"lambda": lambda_coeff, "rank": rank, "epochs": num_epochs},
        "accuracy": None,
        "auc": None,
        "error": "evaluation failed",
    }


def main():
    parser = argparse.ArgumentParser(description="Batch SMP-LoRA sweep")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--epochs", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoints")
    args = parser.parse_args()

    results = []
    total_runs = len(args.lambdas) * len(args.ranks) * len(args.epochs)
    run_id = 0

    print(f"\n{'#'*60}")
    print(f"# SMP-LoRA Hyperparameter Sweep")
    print(f"# Lambdas: {args.lambdas}")
    print(f"# Ranks: {args.ranks}")
    print(f"# Epochs: {args.epochs}")
    print(f"# Total runs: {total_runs}")
    print(f"#{'#'*60}")

    start_time = datetime.now()

    for lambda_coeff in args.lambdas:
        for rank in args.ranks:
            for num_epochs in args.epochs:
                run_id += 1
                name = f"lambda{lambda_coeff}_rank{rank}_ep{num_epochs}"

                print(f"\n\n{'#'*60}")
                print(f"# Run {run_id}/{total_runs}: {name}")
                print(f"#{'#'*60}")

                lora_checkpoint = train_smp_lora(name, lambda_coeff, rank, num_epochs)

                if lora_checkpoint and lora_checkpoint.exists():
                    result = evaluate_smp_lora(name, lora_checkpoint, lambda_coeff, rank, num_epochs)
                    results.append(result)
                    print(f"\nResult: Accuracy={result.get('accuracy')}, AUC={result.get('auc')}")
                else:
                    print(f"Training failed for {name}")
                    results.append({
                        "config": {"lambda": lambda_coeff, "rank": rank, "epochs": num_epochs},
                        "accuracy": None,
                        "auc": None,
                        "error": "training failed",
                    })

                elapsed = (datetime.now() - start_time).total_seconds()
                avg_per_run = elapsed / run_id
                remaining = avg_per_run * (total_runs - run_id)
                print(f"\nElapsed: {elapsed/3600:.1f}h, Est. remaining: {remaining/3600:.1f}h")

    results_file = Path(OUTPUT_BASE) / "sweep_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "results": results,
            "params": {
                "lambdas": args.lambdas,
                "ranks": args.ranks,
                "epochs": args.epochs,
            },
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\n\n{'#'*60}")
    print(f"# Sweep Complete!")
    print(f"# Results saved to: {results_file}")
    print(f"#{'#'*60}")

    valid_results = [r for r in results if r.get("auc") is not None]
    if valid_results:
        valid_results.sort(key=lambda x: x["auc"])
        print("\nResults sorted by AUC (lower is better for defense):")
        for r in valid_results:
            cfg = r["config"]
            print(f"  lambda={cfg['lambda']}, rank={cfg['rank']}, ep={cfg['epochs']}: AUC={r['auc']:.4f}, Acc={r['accuracy']:.4f}")

    return results


if __name__ == "__main__":
    main()
