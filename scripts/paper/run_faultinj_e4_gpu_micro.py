#!/usr/bin/env python3
"""faultinj-e4 GPU micro-demo: short DDPM train with invalid membership GT.

Exploratory only. Labels: faultinj-e4-*. Never writes corrected-s* formal dirs.
Does not change 0.55 / formal hold / claim ceiling.

Design (e4-e5-exploratory-design):
  - Train short on FULL CIFAR-10 train (50k) — invalid contract surface
  - Score with H1-like LR on activations? (expensive) OR use simple loss-based
    membership scores as micro instrument for pathology demo
  - Compare to member-only 25k short control

This micro-demo uses denoising-loss scores as a cheap instrument (not confirmatory H1)
to ask whether invalid GT + full-train short model manufactures higher AUC than
member-only short control under held-out fit/score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "ddpm-cifar10"))
from diffusion import GaussianDiffusionTrainer  # noqa: E402
from model_unet import UNet  # noqa: E402


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Avoid cuDNN stream-mismatch crashes seen on longer micro runs (4070 laptop).
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def build_unet(device: torch.device) -> UNet:
    # slightly smaller for micro budget while staying DDPM-family
    return UNet(
        T=1000,
        ch=64,
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=2,
        dropout=0.1,
    ).to(device)


def train_short(
    loader: DataLoader,
    steps: int,
    device: torch.device,
    seed: int,
    lr: float = 2e-4,
) -> UNet:
    set_seed(seed)
    model = build_unet(device)
    trainer = GaussianDiffusionTrainer(model, 1e-4, 0.02, 1000).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    it = iter(loader)
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device, non_blocking=False)
        loss = trainer(x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % max(1, steps // 5) == 0 or step == steps:
            print(f"  step {step}/{steps} loss={float(loss):.4f}", flush=True)
        if device.type == "cuda" and step % 200 == 0:
            torch.cuda.synchronize()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return model


@torch.no_grad()
def loss_features(
    model: UNet,
    images: torch.Tensor,
    device: torch.device,
    n_t: int = 4,
    seed: int = 0,
    batch_size: int = 128,
) -> np.ndarray:
    """Per-image multi-timestep denoising MSE features (cheap instrument)."""
    model.eval()
    trainer = GaussianDiffusionTrainer(model, 1e-4, 0.02, 1000).to(device)
    # CPU generator avoids device-stream coupling issues with cuDNN.
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x_all = images
    N = x_all.size(0)
    ts = torch.linspace(100, 800, n_t).long()
    out = np.zeros((N, n_t), dtype=np.float32)
    for start in range(0, N, batch_size):
        xb = x_all[start : start + batch_size].to(device, non_blocking=False)
        B = xb.size(0)
        for j, t_scalar in enumerate(ts.tolist()):
            t = torch.full((B,), int(t_scalar), device=device, dtype=torch.long)
            noise = torch.randn(xb.shape, generator=g, device="cpu").to(device)
            betas = trainer.betas
            alphas_bar = torch.cumprod(1.0 - betas, dim=0)
            sqrt_ab = torch.sqrt(alphas_bar[t]).view(B, 1, 1, 1).float()
            sqrt_om = torch.sqrt(1.0 - alphas_bar[t]).view(B, 1, 1, 1).float()
            x_t = sqrt_ab * xb + sqrt_om * noise
            pred = model(x_t, t)
            mse = F.mse_loss(pred, noise, reduction="none").mean(dim=(1, 2, 3))
            out[start : start + B, j] = mse.detach().float().cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
    return out


def heldout_auc(x_cal, y_cal, x_eval, y_eval) -> float:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs", random_state=0),
    )
    clf.fit(x_cal, y_cal)
    scores = clf.decision_function(x_eval)
    # lower loss often => member; flip scores so higher=member if needed by checking direction
    auc = roc_auc_score(y_eval, scores)
    auc_flip = roc_auc_score(y_eval, -scores)
    return float(max(auc, auc_flip))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260718)
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument(
        "--receipt-name",
        type=str,
        default="faultinj_e4_gpu_micro_receipt.json",
        help="receipt filename under out-dir (avoid overwriting prior step budgets)",
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=None,  # dataset path is required; pass --dataset explicitly
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/paper1-corrected-evidence/fault-injection"),
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} steps={args.steps}", flush=True)
    t0 = time.time()
    set_seed(args.seed)

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    # torchvision CIFAR10 root expects parent of cifar-10-batches-py
    ds = datasets.CIFAR10(root=str(args.dataset_root), train=True, download=False, transform=tfm)
    assert len(ds) == 50000

    rng = np.random.default_rng(args.seed)
    all_idx = np.arange(50000)
    member_only = rng.choice(all_idx, size=25000, replace=False)
    member_only.sort()

    # invalid GT evaluation rows: sample 512 from train as "members" and 512 as "nonmembers"
    # but model saw ALL 50k. Half of the "nonmembers" are actually trained indices (pathology).
    eval_member = rng.choice(all_idx, size=args.n_eval, replace=False)
    remaining = np.setdiff1d(all_idx, eval_member)
    eval_nonmember_true = rng.choice(remaining, size=args.n_eval // 2, replace=False)
    # false nonmembers: trained indices labeled nonmember
    eval_nonmember_false = rng.choice(
        np.setdiff1d(remaining, eval_nonmember_true), size=args.n_eval // 2, replace=False
    )
    eval_nonmember = np.concatenate([eval_nonmember_true, eval_nonmember_false])
    # cal split similarly
    pool = np.setdiff1d(all_idx, np.concatenate([eval_member, eval_nonmember]))
    cal_member = rng.choice(pool, size=args.n_eval, replace=False)
    pool2 = np.setdiff1d(pool, cal_member)
    cal_nm_true = rng.choice(pool2, size=args.n_eval // 2, replace=False)
    cal_nm_false = rng.choice(np.setdiff1d(pool2, cal_nm_true), size=args.n_eval // 2, replace=False)
    cal_nonmember = np.concatenate([cal_nm_true, cal_nm_false])

    def make_loader(indices):
        return DataLoader(
            Subset(ds, indices.tolist()),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

    print("== train FULL 50k (invalid-contract surface) ==", flush=True)
    model_full = train_short(make_loader(all_idx), args.steps, device, args.seed)
    print("== train MEMBER-ONLY 25k control ==", flush=True)
    model_mo = train_short(make_loader(member_only), args.steps, device, args.seed + 1)

    def gather(model, members, nonmembers, seed):
        # load images
        def load_idx(idxs):
            xs = torch.stack([ds[i][0] for i in idxs], dim=0)
            return xs

        xm, xn = load_idx(members), load_idx(nonmembers)
        fm = loss_features(model, xm, device, seed=seed)
        fn = loss_features(model, xn, device, seed=seed + 1)
        X = np.concatenate([fm, fn], axis=0)
        y = np.array([1] * len(members) + [0] * len(nonmembers))
        return X, y

    # Invalid-label condition: y uses polluted nonmember definition (includes trained false nonmembers)
    Xc_i, yc_i = gather(model_full, cal_member, cal_nonmember, 11)
    Xe_i, ye_i = gather(model_full, eval_member, eval_nonmember, 12)
    auc_invalid = heldout_auc(Xc_i, yc_i, Xe_i, ye_i)

    # Control: member-only model; evaluate with clean held-out nonmembers only (true non-train indices)
    # Build clean nonmembers from indices outside member_only
    non_train = np.setdiff1d(all_idx, member_only)
    cal_m_c = rng.choice(member_only, size=args.n_eval, replace=False)
    cal_n_c = rng.choice(non_train, size=args.n_eval, replace=False)
    eval_m_c = rng.choice(np.setdiff1d(member_only, cal_m_c), size=args.n_eval, replace=False)
    eval_n_c = rng.choice(np.setdiff1d(non_train, cal_n_c), size=args.n_eval, replace=False)
    Xc_c, yc_c = gather(model_mo, cal_m_c, cal_n_c, 21)
    Xe_c, ye_c = gather(model_mo, eval_m_c, eval_n_c, 22)
    auc_control = heldout_auc(Xc_c, yc_c, Xe_c, ye_c)

    e4_p3 = bool(auc_invalid >= 0.55 and auc_control < 0.55)
    receipt = {
        "protocol_label": "faultinj-e4-gpu-micro-invalid-gt-2026-07-18",
        "evidence_class": "exploratory_non_confirmatory",
        "run_labels": [
            "faultinj-e4-invalid-gt-fulltrain-short",
            "faultinj-e4-member-only-control-short",
        ],
        "created_at_unix": int(time.time()),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "instrument": "multi-timestep denoising MSE features + held-out LR (not confirmatory H1)",
        "auc_invalid_gt_fulltrain": auc_invalid,
        "auc_member_only_control": auc_control,
        "delta_invalid_minus_control": auc_invalid - auc_control,
        "E4_P3_pathology_mark": e4_p3,
        "formal_hold": True,
        "claim_ceiling_unchanged": True,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    receipt["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / args.receipt_name
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    # also save tiny ckpts optional? skip to save disk
    print(json.dumps({"wrote": str(out).replace("\\", "/"), **{k: receipt[k] for k in [
        "auc_invalid_gt_fulltrain","auc_member_only_control","E4_P3_pathology_mark","elapsed_seconds","receipt_sha256"
    ]}}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
