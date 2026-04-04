"""Command-line interface for research audit workflows."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from diffaudit.attacks.secmi import build_secmi_plan
from diffaudit.attacks.secmi_adapter import (
    bootstrap_secmi_smoke_assets,
    prepare_secmi_adapter,
    probe_secmi_dry_run,
    probe_secmi_runtime,
    run_secmi_dry_run,
    summarize_secmi_adapter,
)
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

    secmi_parser = subparsers.add_parser(
        "plan-secmi",
        help="build a SecMI integration plan from audit config",
    )
    secmi_parser.add_argument("--config", required=True, help="path to audit yaml")

    prepare_parser = subparsers.add_parser(
        "prepare-secmi",
        help="prepare adapter context for a local SecMI workspace",
    )
    prepare_parser.add_argument("--config", required=True, help="path to audit yaml")
    prepare_parser.add_argument(
        "--repo-root",
        default="third_party/secmi",
        help="path to vendored or local SecMI repository root",
    )

    dry_run_parser = subparsers.add_parser(
        "dry-run-secmi",
        help="validate SecMI adapter readiness without executing the attack",
    )
    dry_run_parser.add_argument("--config", required=True, help="path to audit yaml")
    dry_run_parser.add_argument(
        "--repo-root",
        default="third_party/secmi",
        help="path to vendored or local SecMI repository root",
    )

    runtime_probe_parser = subparsers.add_parser(
        "runtime-probe-secmi",
        help="validate SecMI runtime readiness by loading flags and model",
    )
    runtime_probe_parser.add_argument("--config", required=True, help="path to audit yaml")
    runtime_probe_parser.add_argument(
        "--repo-root",
        default="third_party/secmi",
        help="path to vendored or local SecMI repository root",
    )

    bootstrap_parser = subparsers.add_parser(
        "bootstrap-secmi-smoke-assets",
        help="create synthetic SecMI smoke assets for local runtime probes",
    )
    bootstrap_parser.add_argument(
        "--target-dir",
        required=True,
        help="directory where smoke assets should be written",
    )
    bootstrap_parser.add_argument(
        "--flagfile-source",
        default="external/SecMI/config/CIFAR10.txt",
        help="path to the reference SecMI flagfile template",
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

    if args.command == "plan-secmi":
        config = load_audit_config(args.config)
        plan = build_secmi_plan(config)
        print(json.dumps(asdict(plan), indent=2, ensure_ascii=True))
        return 0

    if args.command == "prepare-secmi":
        config = load_audit_config(args.config)
        context = prepare_secmi_adapter(config, args.repo_root)
        print(json.dumps(summarize_secmi_adapter(context), indent=2, ensure_ascii=True))
        return 0

    if args.command == "dry-run-secmi":
        config = load_audit_config(args.config)
        exit_code, payload = probe_secmi_dry_run(config, args.repo_root)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return exit_code

    if args.command == "runtime-probe-secmi":
        config = load_audit_config(args.config)
        exit_code, payload = probe_secmi_runtime(config, args.repo_root)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return exit_code

    if args.command == "bootstrap-secmi-smoke-assets":
        payload = bootstrap_secmi_smoke_assets(args.target_dir, args.flagfile_source)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
