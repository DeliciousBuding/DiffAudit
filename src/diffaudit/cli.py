"""Command-line interface for research audit workflows."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from diffaudit.attacks.clid import (
    build_clid_plan,
    explain_clid_assets,
    probe_clid_dry_run,
    run_clid_dry_run_smoke,
)
from diffaudit.attacks.pia import build_pia_plan, explain_pia_assets, probe_pia_dry_run
from diffaudit.attacks.secmi import build_secmi_plan, explain_secmi_assets
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

    pia_parser = subparsers.add_parser(
        "plan-pia",
        help="build a PIA integration plan from audit config",
    )
    pia_parser.add_argument("--config", required=True, help="path to audit yaml")

    clid_parser = subparsers.add_parser(
        "plan-clid",
        help="build a CLiD integration plan from audit config",
    )
    clid_parser.add_argument("--config", required=True, help="path to audit yaml")

    asset_probe_parser = subparsers.add_parser(
        "probe-secmi-assets",
        help="inspect SecMI asset readiness without importing the full runtime",
    )
    asset_probe_parser.add_argument("--config", required=True, help="path to audit yaml")
    asset_probe_parser.add_argument(
        "--member-split-root",
        default="third_party/secmi/mia_evals/member_splits",
        help="path to SecMI member split npz files",
    )

    pia_asset_probe_parser = subparsers.add_parser(
        "probe-pia-assets",
        help="inspect PIA DDPM asset readiness without importing the full runtime",
    )
    pia_asset_probe_parser.add_argument("--config", required=True, help="path to audit yaml")
    pia_asset_probe_parser.add_argument(
        "--member-split-root",
        default="external/PIA/DDPM",
        help="path to PIA DDPM member split npz files",
    )

    clid_asset_probe_parser = subparsers.add_parser(
        "probe-clid-assets",
        help="inspect CLiD asset readiness without importing the runtime",
    )
    clid_asset_probe_parser.add_argument("--config", required=True, help="path to audit yaml")

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

    pia_dry_run_parser = subparsers.add_parser(
        "dry-run-pia",
        help="validate PIA DDPM readiness without executing the attack",
    )
    pia_dry_run_parser.add_argument("--config", required=True, help="path to audit yaml")
    pia_dry_run_parser.add_argument(
        "--repo-root",
        default="external/PIA",
        help="path to local PIA repository root",
    )
    pia_dry_run_parser.add_argument(
        "--member-split-root",
        default="external/PIA/DDPM",
        help="path to PIA DDPM member split npz files",
    )

    clid_dry_run_parser = subparsers.add_parser(
        "dry-run-clid",
        help="validate CLiD readiness without executing the attack",
    )
    clid_dry_run_parser.add_argument("--config", required=True, help="path to audit yaml")
    clid_dry_run_parser.add_argument(
        "--repo-root",
        default="external/CLiD",
        help="path to local CLiD repository root",
    )

    clid_dry_run_smoke_parser = subparsers.add_parser(
        "run-clid-dry-run-smoke",
        help="run a synthetic dry-run smoke for CLiD",
    )
    clid_dry_run_smoke_parser.add_argument(
        "--workspace",
        required=True,
        help="workspace directory for CLiD dry-run smoke artifacts",
    )
    clid_dry_run_smoke_parser.add_argument(
        "--repo-root",
        default="external/CLiD",
        help="path to local CLiD repository root",
    )

    pia_runtime_probe_parser = subparsers.add_parser(
        "runtime-probe-pia",
        help="validate PIA runtime readiness by loading modules, checkpoint and attacker",
    )
    pia_runtime_probe_parser.add_argument("--config", required=True, help="path to audit yaml")
    pia_runtime_probe_parser.add_argument(
        "--repo-root",
        default="external/PIA",
        help="path to local PIA repository root",
    )
    pia_runtime_probe_parser.add_argument(
        "--member-split-root",
        default="external/PIA/DDPM",
        help="path to PIA DDPM member split npz files",
    )
    pia_runtime_probe_parser.add_argument(
        "--device",
        default="cpu",
        help="device used for the runtime probe",
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

    pia_bootstrap_parser = subparsers.add_parser(
        "bootstrap-pia-smoke-assets",
        help="create synthetic PIA smoke assets for local runtime probes",
    )
    pia_bootstrap_parser.add_argument(
        "--target-dir",
        required=True,
        help="directory where smoke assets should be written",
    )
    pia_bootstrap_parser.add_argument(
        "--repo-root",
        default="external/PIA",
        help="path to local PIA repository root",
    )

    synth_smoke_parser = subparsers.add_parser(
        "run-secmi-synth-smoke",
        help="run a synthetic SecMI stat smoke execution",
    )
    synth_smoke_parser.add_argument(
        "--workspace",
        required=True,
        help="workspace directory for synthetic smoke artifacts",
    )
    synth_smoke_parser.add_argument(
        "--device",
        default="cpu",
        help="device used for the synthetic smoke run",
    )

    pia_synth_smoke_parser = subparsers.add_parser(
        "run-pia-synth-smoke",
        help="run a synthetic PIA smoke execution",
    )
    pia_synth_smoke_parser.add_argument(
        "--workspace",
        required=True,
        help="workspace directory for synthetic smoke artifacts",
    )
    pia_synth_smoke_parser.add_argument(
        "--repo-root",
        default="external/PIA",
        help="path to local PIA repository root",
    )
    pia_synth_smoke_parser.add_argument(
        "--device",
        default="cpu",
        help="device used for the synthetic smoke run",
    )

    pia_runtime_smoke_parser = subparsers.add_parser(
        "run-pia-runtime-smoke",
        help="run a config-driven PIA runtime smoke with synthetic assets",
    )
    pia_runtime_smoke_parser.add_argument(
        "--workspace",
        required=True,
        help="workspace directory for runtime smoke artifacts",
    )
    pia_runtime_smoke_parser.add_argument(
        "--repo-root",
        default="external/PIA",
        help="path to local PIA repository root",
    )
    pia_runtime_smoke_parser.add_argument(
        "--device",
        default="cpu",
        help="device used for the runtime smoke run",
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

    if args.command == "plan-pia":
        config = load_audit_config(args.config)
        plan = build_pia_plan(config)
        print(json.dumps(asdict(plan), indent=2, ensure_ascii=True))
        return 0

    if args.command == "plan-clid":
        config = load_audit_config(args.config)
        plan = build_clid_plan(config)
        print(json.dumps(asdict(plan), indent=2, ensure_ascii=True))
        return 0

    if args.command == "probe-secmi-assets":
        config = load_audit_config(args.config)
        payload = explain_secmi_assets(config, member_split_root=args.member_split_root)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0 if payload["status"] == "ready" else 1

    if args.command == "probe-pia-assets":
        config = load_audit_config(args.config)
        payload = explain_pia_assets(config, member_split_root=args.member_split_root)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0 if payload["status"] == "ready" else 1

    if args.command == "probe-clid-assets":
        config = load_audit_config(args.config)
        payload = explain_clid_assets(config)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0 if payload["status"] == "ready" else 1

    if args.command == "prepare-secmi":
        from diffaudit.attacks.secmi_adapter import prepare_secmi_adapter, summarize_secmi_adapter

        config = load_audit_config(args.config)
        context = prepare_secmi_adapter(config, args.repo_root)
        print(json.dumps(summarize_secmi_adapter(context), indent=2, ensure_ascii=True))
        return 0

    if args.command == "dry-run-secmi":
        from diffaudit.attacks.secmi_adapter import probe_secmi_dry_run

        config = load_audit_config(args.config)
        exit_code, payload = probe_secmi_dry_run(config, args.repo_root)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return exit_code

    if args.command == "dry-run-pia":
        config = load_audit_config(args.config)
        exit_code, payload = probe_pia_dry_run(
            config,
            args.repo_root,
            member_split_root=args.member_split_root,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return exit_code

    if args.command == "dry-run-clid":
        config = load_audit_config(args.config)
        exit_code, payload = probe_clid_dry_run(config, args.repo_root)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return exit_code

    if args.command == "run-clid-dry-run-smoke":
        payload = run_clid_dry_run_smoke(
            args.workspace,
            repo_root=args.repo_root,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0 if payload["status"] == "ready" else 1

    if args.command == "runtime-probe-pia":
        from diffaudit.attacks.pia_adapter import probe_pia_runtime

        config = load_audit_config(args.config)
        exit_code, payload = probe_pia_runtime(
            config,
            args.repo_root,
            member_split_root=args.member_split_root,
            device=args.device,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return exit_code

    if args.command == "runtime-probe-secmi":
        from diffaudit.attacks.secmi_adapter import probe_secmi_runtime

        config = load_audit_config(args.config)
        exit_code, payload = probe_secmi_runtime(config, args.repo_root)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return exit_code

    if args.command == "bootstrap-secmi-smoke-assets":
        from diffaudit.attacks.secmi_adapter import bootstrap_secmi_smoke_assets

        payload = bootstrap_secmi_smoke_assets(args.target_dir, args.flagfile_source)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0

    if args.command == "bootstrap-pia-smoke-assets":
        from diffaudit.attacks.pia_adapter import bootstrap_pia_smoke_assets

        payload = bootstrap_pia_smoke_assets(args.target_dir, args.repo_root)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0

    if args.command == "run-secmi-synth-smoke":
        from diffaudit.attacks.secmi_adapter import run_synthetic_secmi_stat_smoke

        payload = run_synthetic_secmi_stat_smoke(args.workspace, device=args.device)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0

    if args.command == "run-pia-synth-smoke":
        from diffaudit.attacks.pia_adapter import run_synthetic_pia_smoke

        payload = run_synthetic_pia_smoke(
            args.workspace,
            repo_root=args.repo_root,
            device=args.device,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0

    if args.command == "run-pia-runtime-smoke":
        from diffaudit.attacks.pia_adapter import run_pia_runtime_smoke

        payload = run_pia_runtime_smoke(
            args.workspace,
            repo_root=args.repo_root,
            device=args.device,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0 if payload["status"] == "ready" else 1

    parser.error(f"Unsupported command: {args.command}")
    return 2
