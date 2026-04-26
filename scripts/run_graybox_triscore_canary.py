from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from diffaudit.attacks.graybox_triscore import run_graybox_triscore_canary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the X-88 gray-box tri-score offline canary.")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Optional explicit run root. Defaults to the canonical X-88 timestamped gray-box run directory.",
    )
    parser.add_argument(
        "--contract-reference",
        type=Path,
        default=Path("workspaces/gray-box/runs/cdi-paired-canary-20260417-r3-contract/audit_summary.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = args.run_root or Path(f"workspaces/gray-box/runs/x88-cdi-tmiadm-triscore-canary-{timestamp}")
    summary = run_graybox_triscore_canary(
        run_root=run_root,
        contract_reference_path=args.contract_reference,
        surfaces=[
            {
                "name": "gpu256_undefended",
                "pia_scores_path": Path("workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-256/scores.json"),
                "pia_summary_path": Path("workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260408-gpu-256/summary.json"),
                "tmiadm_family_scores_path": Path(
                    "workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-256-r2-seed1/family-scores.json"
                ),
                "tmiadm_summary_path": Path(
                    "workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-256-r2-seed1/summary.json"
                ),
                "tmiadm_family": "long_window",
            },
            {
                "name": "gpu256_defended",
                "pia_scores_path": Path(
                    "workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-256/scores.json"
                ),
                "pia_summary_path": Path(
                    "workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260408-gpu-256/summary.json"
                ),
                "tmiadm_family_scores_path": Path(
                    "workspaces/gray-box/runs/tmiadm-cifar10-late-window-temporal-striding-defense-20260416-gpu-256-r2-seed1/family-scores.json"
                ),
                "tmiadm_summary_path": Path(
                    "workspaces/gray-box/runs/tmiadm-cifar10-late-window-temporal-striding-defense-20260416-gpu-256-r2-seed1/summary.json"
                ),
                "tmiadm_family": "long_window",
            },
        ],
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
