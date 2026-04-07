"""Run the repository's standard local quality gates."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    completed = subprocess.run(cmd, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fast",
        action="store_true",
        help="run the minimal fast check suite",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    run([sys.executable, "scripts/bootstrap_research_env.py"], repo_root)
    run([sys.executable, "-m", "diffaudit", "--help"], repo_root)
    run([sys.executable, "-m", "unittest", "tests.test_attack_registry", "tests.test_smoke_pipeline"], repo_root)
    run([sys.executable, "-m", "pytest", "tests/test_variation_attack.py", "tests/test_render_team_local_configs.py", "-q"], repo_root)

    if not args.fast:
        run(
            [
                sys.executable,
                "scripts/render_team_local_configs.py",
                "--team-local",
                "configs/assets/team.local.template.yaml",
                "--output-dir",
                "tmp/configs/rendered-checks",
            ],
            repo_root,
        )


if __name__ == "__main__":
    main()
