"""Batch evaluation of all SMP-LoRA sweep checkpoints."""

import json
import subprocess
import sys
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = str(RESEARCH_ROOT / "workspaces" / "white-box" / "assets" / "gsa-gpu-128" / "checkpoints" / "target" / "checkpoint-64")
MEMBER_DIR = str(RESEARCH_ROOT / "workspaces" / "white-box" / "assets" / "gsa-cifar10-1k-3shadow" / "datasets" / "target-member")
NONMEMBER_DIR = str(RESEARCH_ROOT / "workspaces" / "white-box" / "assets" / "gsa-cifar10-1k-3shadow" / "datasets" / "target-nonmember")
SWEEP_DIR = RESEARCH_ROOT / "outputs" / "smp-lora-sweep"


def main():
    results = []

    for config_dir in sorted(SWEEP_DIR.iterdir()):
        if not config_dir.is_dir():
            continue

        lora_checkpoint = config_dir / "final" / "lora_weights.pt"
        if not lora_checkpoint.exists():
            print(f"Skip {config_dir.name}: no final checkpoint")
            continue

        eval_output = config_dir / "evaluation_pretrained.json"
        if eval_output.exists():
            with open(eval_output) as f:
                existing = json.load(f)
            if existing.get("auc") is not None:
                print(f"Skip {config_dir.name}: already evaluated (AUC={existing['auc']:.4f})")
                results.append({"name": config_dir.name, **existing})
                continue

        print(f"\nEvaluating {config_dir.name}...")

        cmd = [
            "conda", "run", "-n", "diffaudit-research",
            "python", str(RESEARCH_ROOT / "scripts" / "evaluate_smp_lora_defense.py"),
            "--lora_checkpoint", str(lora_checkpoint),
            "--base_model", BASE_MODEL,
            "--member_dir", MEMBER_DIR,
            "--nonmember_dir", NONMEMBER_DIR,
            "--device", "cuda",
            "--num_samples", "500",
            "--output", str(eval_output),
        ]

        import subprocess
        result = subprocess.run(cmd, timeout=600, capture_output=True, text=True)
        print(f"  exit={result.returncode}")

        if result.returncode != 0:
            print(f"  STDERR: {result.stderr[-300:]}")

        if eval_output.exists():
            with open(eval_output) as f:
                eval_result = json.load(f)
            results.append({"name": config_dir.name, **eval_result})
            print(f"  AUC={eval_result.get('auc')}, Acc={eval_result.get('accuracy')}")
        else:
            results.append({"name": config_dir.name, "auc": None, "accuracy": None})

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")

    valid = [r for r in results if r.get("auc") is not None]
    valid.sort(key=lambda x: x["auc"])

    for r in valid:
        print(f"  {r['name']:30s} | AUC={r['auc']:.4f} | Acc={r['accuracy']:.4f}")

    failed = [r for r in results if r.get("auc") is None]
    if failed:
        print(f"\nFailed evaluations: {len(failed)}")
        for r in failed:
            print(f"  {r['name']}")

    results_file = SWEEP_DIR / "eval_summary.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {results_file}")


if __name__ == "__main__":
    main()
