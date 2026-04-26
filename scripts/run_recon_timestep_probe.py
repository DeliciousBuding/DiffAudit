from __future__ import annotations

import argparse
import json
from pathlib import Path

from diffaudit.attacks.recon_timestep_probe import analyze_recon_timestep_probe, load_recon_vector_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe timestep/coordinate-selective signal in existing recon score artifacts.")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)

    shadow_member, _ = load_recon_vector_artifact(args.artifact_dir / "shadow_member.pt")
    shadow_nonmember, _ = load_recon_vector_artifact(args.artifact_dir / "shadow_non_member.pt")
    target_member, _ = load_recon_vector_artifact(args.artifact_dir / "target_member.pt")
    target_nonmember, _ = load_recon_vector_artifact(args.artifact_dir / "target_non_member.pt")

    result = analyze_recon_timestep_probe(
        shadow_member=shadow_member,
        shadow_nonmember=shadow_nonmember,
        target_member=target_member,
        target_nonmember=target_nonmember,
    )
    summary = {
        "task_scope": "BB-1.3 timestep-selective reconstruction probe",
        "track": "black-box",
        "method_family": "recon-coordinate-selective",
        "artifact_dir": args.artifact_dir.as_posix(),
        **result,
    }
    (args.run_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
