from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SECMI_ROOT = ROOT / "external" / "SecMI"
if str(SECMI_ROOT) not in sys.path:
    sys.path.insert(0, str(SECMI_ROOT))

SECMI_EVALS = ROOT / "external" / "SecMI" / "mia_evals"
if str(SECMI_EVALS) not in sys.path:
    sys.path.insert(0, str(SECMI_EVALS))

from dataset_utils import load_member_data
from diffaudit.attacks.secmi_adapter import load_vendored_secmi_module, parse_secmi_flagfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded SecMI-vs-PIA disagreement analysis.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--flagfile-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--member-split-root", type=Path, required=True)
    parser.add_argument("--pia-scores", type=Path, required=True)
    parser.add_argument("--subset-limit", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--t-sec", type=int, default=100)
    parser.add_argument("--timestep", type=int, default=10)
    return parser.parse_args()


def orient_memberness(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    if float(member_scores.mean()) < float(nonmember_scores.mean()):
        return -member_scores, -nonmember_scores, "negated"
    return member_scores, nonmember_scores, "identity"


def rank_top_overlap(scores_a: np.ndarray, scores_b: np.ndarray, k: int) -> float:
    idx_a = set(np.argsort(scores_a)[-k:].tolist())
    idx_b = set(np.argsort(scores_b)[-k:].tolist())
    return len(idx_a & idx_b) / float(k)


def main() -> None:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    start = time.time()

    flags = parse_secmi_flagfile(args.flagfile_path)
    secmi = load_vendored_secmi_module()
    model = secmi.get_model(str(args.checkpoint_path), flags, WA=True).cuda()
    model.eval()

    member_set, nonmember_set, _, _ = load_member_data(
        dataset_root=str(args.dataset_root),
        dataset_name="cifar10",
        batch_size=args.batch_size,
        member_split_root=str(args.member_split_root),
        shuffle=False,
        randaugment=False,
    )
    member_subset = torch.utils.data.Subset(member_set, list(range(args.subset_limit)))
    nonmember_subset = torch.utils.data.Subset(nonmember_set, list(range(args.subset_limit)))
    member_loader = torch.utils.data.DataLoader(member_subset, batch_size=args.batch_size, shuffle=False)
    nonmember_loader = torch.utils.data.DataLoader(nonmember_subset, batch_size=args.batch_size, shuffle=False)

    member_results = secmi.get_intermediate_results(model, flags, member_loader, t_sec=args.t_sec, timestep=args.timestep)
    nonmember_results = secmi.get_intermediate_results(model, flags, nonmember_loader, t_sec=args.t_sec, timestep=args.timestep)
    attack_results = secmi.execute_attack(
        {
            "member_diffusions": member_results["internal_diffusions"],
            "member_internal_samples": member_results["internal_denoise"],
            "nonmember_diffusions": nonmember_results["internal_diffusions"],
            "nonmember_internal_samples": nonmember_results["internal_denoise"],
        },
        type="stat",
    )

    secmi_member_raw = attack_results["member_scores"].detach().cpu().numpy().reshape(-1)[: args.subset_limit]
    secmi_nonmember_raw = attack_results["nonmember_scores"].detach().cpu().numpy().reshape(-1)[: args.subset_limit]
    secmi_memberness_member, secmi_memberness_nonmember, secmi_orientation = orient_memberness(
        secmi_member_raw, secmi_nonmember_raw
    )

    pia_payload = json.loads(args.pia_scores.read_text(encoding="utf-8"))
    pia_member_raw = np.asarray(pia_payload["member_scores"][: args.subset_limit], dtype=float)
    pia_nonmember_raw = np.asarray(pia_payload["nonmember_scores"][: args.subset_limit], dtype=float)
    pia_memberness_member, pia_memberness_nonmember, pia_orientation = orient_memberness(
        pia_member_raw, pia_nonmember_raw
    )

    combined_secmi = np.concatenate([secmi_memberness_member, secmi_memberness_nonmember])
    combined_pia = np.concatenate([pia_memberness_member, pia_memberness_nonmember])
    labels = np.concatenate([np.ones(args.subset_limit), np.zeros(args.subset_limit)])

    member_corr = float(spearmanr(secmi_memberness_member, pia_memberness_member).statistic)
    nonmember_corr = float(spearmanr(secmi_memberness_nonmember, pia_memberness_nonmember).statistic)
    combined_corr = float(spearmanr(combined_secmi, combined_pia).statistic)
    topk = max(16, args.subset_limit // 10)
    overlap = rank_top_overlap(combined_secmi, combined_pia, topk)

    disagreement_mask = ((combined_secmi >= np.median(combined_secmi)) != (combined_pia >= np.median(combined_pia)))
    disagreement_rate = float(disagreement_mask.mean())

    summary = {
        "run_id": args.run_root.name,
        "track": "gray-box",
        "method": "secmi-pia-disagreement",
        "dataset": "CIFAR-10",
        "mode": "bounded-real-asset-disagreement-variant",
        "status": "ready",
        "assets": {
            "checkpoint_path": str(args.checkpoint_path.as_posix()),
            "flagfile_path": str(args.flagfile_path.as_posix()),
            "dataset_root": str(args.dataset_root.as_posix()),
            "member_split_root": str(args.member_split_root.as_posix()),
            "flag_config_source": "flagfile",
        },
        "runtime": {
            "score_contract": "secmi-stat-mainline-like",
            "subset_limit": args.subset_limit,
            "batch_size": args.batch_size,
            "t_sec": args.t_sec,
            "timestep": args.timestep,
        },
        "metrics": {
            "secmi_stat_auc": round(float(attack_results["auc"]), 6),
            "secmi_stat_asr": round(float(attack_results["asr"]), 6),
            "member_spearman": round(member_corr, 6),
            "nonmember_spearman": round(nonmember_corr, 6),
            "combined_spearman": round(combined_corr, 6),
            "topk_overlap": round(overlap, 6),
            "disagreement_rate": round(disagreement_rate, 6),
        },
        "analysis": {
            "subset_limit": args.subset_limit,
            "topk": topk,
            "secmi_orientation": secmi_orientation,
            "pia_orientation": pia_orientation,
            "secmi_member_mean_raw": round(float(secmi_member_raw.mean()), 6),
            "secmi_nonmember_mean_raw": round(float(secmi_nonmember_raw.mean()), 6),
            "pia_member_mean_raw": round(float(pia_member_raw.mean()), 6),
            "pia_nonmember_mean_raw": round(float(pia_nonmember_raw.mean()), 6),
        },
        "cost": {
            "gpu_hours": round((time.time() - start) / 3600.0, 4),
        },
        "notes": (
            "Bounded real-asset SecMI stat variant with per-sample score export and disagreement analysis against the "
            "provided PIA score surface."
        ),
        "artifacts": {
            "summary": str((args.run_root / "summary.json").as_posix()),
            "secmi_member_scores": str((args.run_root / "secmi_member_scores.npy").as_posix()),
            "secmi_nonmember_scores": str((args.run_root / "secmi_nonmember_scores.npy").as_posix()),
            "pia_scores_source": str(args.pia_scores.as_posix()),
        },
    }

    np.save(args.run_root / "secmi_member_scores.npy", secmi_member_raw)
    np.save(args.run_root / "secmi_nonmember_scores.npy", secmi_nonmember_raw)
    (args.run_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
