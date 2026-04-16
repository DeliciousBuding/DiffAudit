from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffaudit.attacks.mofit_scaffold import initialize_mofit_scaffold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize a bounded MoFit interface scaffold.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--member-dir", type=Path, required=True)
    parser.add_argument("--nonmember-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--lora-dir", type=Path, required=True)
    parser.add_argument("--blip-dir", type=Path, required=True)
    parser.add_argument("--member-limit", type=int, default=1)
    parser.add_argument("--member-offset", type=int, default=0)
    parser.add_argument("--nonmember-limit", type=int, default=1)
    parser.add_argument("--nonmember-offset", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--surrogate-steps", type=int, default=8)
    parser.add_argument("--embedding-steps", type=int, default=12)
    parser.add_argument("--surrogate-lr", type=float, default=1e-2)
    parser.add_argument("--embedding-lr", type=float, default=5e-3)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--max-timestep", type=int, default=140)
    parser.add_argument("--record-trace", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = initialize_mofit_scaffold(
        run_root=args.run_root,
        target_family="sd15-plus-celeba_partial_target-checkpoint-25000",
        caption_source="metadata-with-blip-fallback",
        surrogate_steps=args.surrogate_steps,
        embedding_steps=args.embedding_steps,
        member_count=args.member_limit,
        nonmember_count=args.nonmember_limit,
    )
    payload.update(
        {
            "status": "scaffold_only",
            "member_dir": args.member_dir.as_posix(),
            "nonmember_dir": args.nonmember_dir.as_posix(),
            "model_dir": args.model_dir.as_posix(),
            "lora_dir": args.lora_dir.as_posix(),
            "blip_dir": args.blip_dir.as_posix(),
            "device": args.device,
            "surrogate_lr": args.surrogate_lr,
            "embedding_lr": args.embedding_lr,
            "guidance_scale": args.guidance_scale,
            "max_timestep": args.max_timestep,
            "record_trace": args.record_trace,
        }
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
