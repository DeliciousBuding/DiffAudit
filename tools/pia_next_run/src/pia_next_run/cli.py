from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .gate import run_gate, write_outputs
from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pia-next-run",
        description="Validate PIA next-run intake and emit manifest+provenance JSON.",
    )
    parser.add_argument("--config", required=True, type=Path, help="PIA runtime config file or directory.")
    parser.add_argument("--member-split-root", required=True, type=Path, help="Directory containing member split files.")
    parser.add_argument("--repo-root", required=True, type=Path, help="Repository root used for git provenance.")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="Output directory for manifest/provenance.")
    parser.add_argument("--strict", action="store_true", help="Fail when repo_root is not clean or not a git repo.")
    parser.add_argument("--stdout", action="store_true", help="Print JSON payload instead of writing files.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    check, manifest, provenance = run_gate(
        config_path=args.config,
        member_split_root=args.member_split_root,
        repo_root=args.repo_root,
        strict=bool(args.strict),
    )
    if args.stdout:
        sys.stdout.write(json.dumps({"manifest": manifest, "provenance": provenance}, indent=2, sort_keys=True) + "\n")
    else:
        manifest_path, provenance_path = write_outputs(args.out_dir, manifest, provenance)
        sys.stdout.write(f"OK={str(check.ok).lower()} manifest={manifest_path} provenance={provenance_path}\n")
        for error in check.errors:
            sys.stdout.write(f"ERROR {error}\n")
        for warning in check.warnings:
            sys.stdout.write(f"WARN {warning}\n")
    return 0 if check.ok else 2
