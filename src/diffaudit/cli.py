"""Command-line interface for research audit workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from diffaudit.config import load_audit_config
from diffaudit.pipelines.smoke import run_smoke_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="diffaudit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke_parser = subparsers.add_parser("run-smoke", help="run the smoke pipeline")
    smoke_parser.add_argument("--config", required=True, help="path to audit yaml")
    smoke_parser.add_argument(
        "--workspace",
        default=".",
        help="workspace root used to resolve output paths",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run-smoke":
        config = load_audit_config(args.config)
        summary_path = run_smoke_pipeline(config, Path(args.workspace))
        print(f"Smoke summary written to {summary_path}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
