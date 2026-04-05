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
    summarize_clid_artifacts,
)
from diffaudit.attacks.dit import probe_dit_assets, run_dit_sample_smoke
from diffaudit.attacks.pia import build_pia_plan, explain_pia_assets, probe_pia_dry_run
from diffaudit.attacks.recon import (
    build_recon_plan,
    explain_recon_assets,
    probe_recon_dry_run,
    probe_recon_runtime_assets,
    probe_recon_score_artifacts,
    run_recon_artifact_mainline,
    run_recon_eval_smoke,
    run_recon_mainline_smoke,
    run_recon_runtime_mainline,
    run_recon_upstream_eval_smoke,
    summarize_recon_artifacts,
)
from diffaudit.attacks.secmi import build_secmi_plan, explain_secmi_assets
from diffaudit.attacks.variation import (
    build_variation_plan,
    explain_variation_assets,
    probe_variation_dry_run,
    run_variation_synthetic_smoke,
)
from diffaudit.config import load_audit_config
from diffaudit.pipelines.smoke import run_smoke_pipeline
from diffaudit.reports.blackbox_status import build_blackbox_status_report


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

    recon_parser = subparsers.add_parser(
        "plan-recon",
        help="build a reconstruction-based black-box integration plan from audit config",
    )
    recon_parser.add_argument("--config", required=True, help="path to audit yaml")

    variation_parser = subparsers.add_parser(
        "plan-variation",
        help="build an API-only variation attack plan from audit config",
    )
    variation_parser.add_argument("--config", required=True, help="path to audit yaml")

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

    recon_asset_probe_parser = subparsers.add_parser(
        "probe-recon-assets",
        help="inspect reconstruction-based black-box asset readiness without importing the runtime",
    )
    recon_asset_probe_parser.add_argument("--config", required=True, help="path to audit yaml")

    recon_score_probe_parser = subparsers.add_parser(
        "probe-recon-score-artifacts",
        help="inspect reconstruction score artifact readiness for artifact-driven mainline runs",
    )
    recon_score_probe_parser.add_argument(
        "--artifact-dir",
        required=True,
        help="directory containing target/shadow member and non-member score artifacts",
    )

    recon_runtime_probe_parser = subparsers.add_parser(
        "probe-recon-runtime-assets",
        help="inspect reconstruction dataset/model assets for runtime artifact generation",
    )
    recon_runtime_probe_parser.add_argument("--target-member-dataset", required=True)
    recon_runtime_probe_parser.add_argument("--target-nonmember-dataset", required=True)
    recon_runtime_probe_parser.add_argument("--shadow-member-dataset", required=True)
    recon_runtime_probe_parser.add_argument("--shadow-nonmember-dataset", required=True)
    recon_runtime_probe_parser.add_argument("--target-model-dir", required=True)
    recon_runtime_probe_parser.add_argument("--shadow-model-dir", required=True)
    recon_runtime_probe_parser.add_argument(
        "--repo-root",
        default="external/Reconstruction-based-Attack",
        help="path to local reconstruction-based attack repository root",
    )

    dit_asset_probe_parser = subparsers.add_parser(
        "probe-dit-assets",
        help="inspect official DiT sampling workspace and optional checkpoint readiness",
    )
    dit_asset_probe_parser.add_argument(
        "--repo-root",
        default="external/DiT",
        help="path to local official DiT repository root",
    )
    dit_asset_probe_parser.add_argument(
        "--model",
        default="DiT-XL/2",
        help="DiT model name passed to the official sample script",
    )
    dit_asset_probe_parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="image size passed to the official sample script",
    )
    dit_asset_probe_parser.add_argument(
        "--ckpt",
        default=None,
        help="optional local DiT checkpoint path; omitted means the official auto-download path",
    )

    variation_asset_probe_parser = subparsers.add_parser(
        "probe-variation-assets",
        help="inspect API-only variation attack readiness without remote calls",
    )
    variation_asset_probe_parser.add_argument("--config", required=True, help="path to audit yaml")

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

    recon_dry_run_parser = subparsers.add_parser(
        "dry-run-recon",
        help="validate reconstruction-based black-box readiness without executing the attack",
    )
    recon_dry_run_parser.add_argument("--config", required=True, help="path to audit yaml")
    recon_dry_run_parser.add_argument(
        "--repo-root",
        default="external/Reconstruction-based-Attack",
        help="path to local reconstruction-based attack repository root",
    )

    variation_dry_run_parser = subparsers.add_parser(
        "dry-run-variation",
        help="validate API-only variation attack readiness without remote calls",
    )
    variation_dry_run_parser.add_argument("--config", required=True, help="path to audit yaml")

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

    clid_artifact_summary_parser = subparsers.add_parser(
        "summarize-clid-artifacts",
        help="summarize upstream CLiD inter_output artifacts into a repository summary",
    )
    clid_artifact_summary_parser.add_argument(
        "--artifact-dir",
        required=True,
        help="directory containing CLiD inter_output artifact txt files",
    )
    clid_artifact_summary_parser.add_argument(
        "--workspace",
        required=True,
        help="workspace directory for the CLiD artifact summary",
    )

    recon_eval_smoke_parser = subparsers.add_parser(
        "run-recon-eval-smoke",
        help="run a synthetic evaluation smoke for the reconstruction-based black-box line",
    )
    recon_eval_smoke_parser.add_argument(
        "--workspace",
        required=True,
        help="workspace directory for reconstruction eval smoke artifacts",
    )

    recon_artifact_summary_parser = subparsers.add_parser(
        "summarize-recon-artifacts",
        help="summarize reconstruction score artifacts into a repository summary",
    )
    recon_artifact_summary_parser.add_argument(
        "--artifact-dir",
        required=True,
        help="directory containing target/shadow member and non-member score artifacts",
    )
    recon_artifact_summary_parser.add_argument(
        "--workspace",
        required=True,
        help="workspace directory for the reconstruction artifact summary",
    )

    recon_upstream_eval_smoke_parser = subparsers.add_parser(
        "run-recon-upstream-eval-smoke",
        help="run the upstream reconstruction evaluation script with synthetic score artifacts",
    )
    recon_upstream_eval_smoke_parser.add_argument(
        "--workspace",
        required=True,
        help="workspace directory for reconstruction upstream eval smoke artifacts",
    )
    recon_upstream_eval_smoke_parser.add_argument(
        "--repo-root",
        default="external/Reconstruction-based-Attack",
        help="path to local reconstruction-based attack repository root",
    )
    recon_upstream_eval_smoke_parser.add_argument(
        "--method",
        default="threshold",
        help="evaluation method passed to the upstream script",
    )

    recon_mainline_smoke_parser = subparsers.add_parser(
        "run-recon-mainline-smoke",
        help="run the unified reconstruction black-box smoke mainline and emit a single summary",
    )
    recon_mainline_smoke_parser.add_argument(
        "--workspace",
        required=True,
        help="workspace directory for reconstruction mainline smoke artifacts",
    )
    recon_mainline_smoke_parser.add_argument(
        "--repo-root",
        default="external/Reconstruction-based-Attack",
        help="path to local reconstruction-based attack repository root",
    )
    recon_mainline_smoke_parser.add_argument(
        "--method",
        default="threshold",
        help="evaluation method passed to the upstream script",
    )

    recon_artifact_mainline_parser = subparsers.add_parser(
        "run-recon-artifact-mainline",
        help="run the reconstruction black-box mainline with provided score artifacts",
    )
    recon_artifact_mainline_parser.add_argument(
        "--artifact-dir",
        required=True,
        help="directory containing target/shadow member and non-member score artifacts",
    )
    recon_artifact_mainline_parser.add_argument(
        "--workspace",
        required=True,
        help="workspace directory for reconstruction artifact mainline outputs",
    )
    recon_artifact_mainline_parser.add_argument(
        "--repo-root",
        default="external/Reconstruction-based-Attack",
        help="path to local reconstruction-based attack repository root",
    )
    recon_artifact_mainline_parser.add_argument(
        "--method",
        default="threshold",
        help="evaluation method passed to the upstream script",
    )

    recon_runtime_mainline_parser = subparsers.add_parser(
        "run-recon-runtime-mainline",
        help="generate reconstruction score artifacts from dataset payloads and run the artifact mainline",
    )
    recon_runtime_mainline_parser.add_argument("--target-member-dataset", required=True)
    recon_runtime_mainline_parser.add_argument("--target-nonmember-dataset", required=True)
    recon_runtime_mainline_parser.add_argument("--shadow-member-dataset", required=True)
    recon_runtime_mainline_parser.add_argument("--shadow-nonmember-dataset", required=True)
    recon_runtime_mainline_parser.add_argument("--target-model-dir", required=True)
    recon_runtime_mainline_parser.add_argument("--shadow-model-dir", required=True)
    recon_runtime_mainline_parser.add_argument(
        "--workspace",
        required=True,
        help="workspace directory for runtime reconstruction artifacts and summaries",
    )
    recon_runtime_mainline_parser.add_argument(
        "--repo-root",
        default="external/Reconstruction-based-Attack",
        help="path to local reconstruction-based attack repository root",
    )
    recon_runtime_mainline_parser.add_argument(
        "--pretrained-model-name-or-path",
        default="runwayml/stable-diffusion-v1-5",
        help="base pretrained model identifier used by the upstream inference script",
    )
    recon_runtime_mainline_parser.add_argument(
        "--num-validation-images",
        type=int,
        default=3,
        help="number of images generated per query sample",
    )
    recon_runtime_mainline_parser.add_argument(
        "--inference-steps",
        type=int,
        default=30,
        help="number of diffusion steps used by the upstream inference script",
    )
    recon_runtime_mainline_parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index passed to the embedding script",
    )
    recon_runtime_mainline_parser.add_argument(
        "--backend",
        default="stable_diffusion",
        help="runtime backend used to generate reconstruction images",
    )
    recon_runtime_mainline_parser.add_argument(
        "--scheduler",
        default="default",
        help="stable diffusion scheduler used by the upstream inference script",
    )
    recon_runtime_mainline_parser.add_argument(
        "--method",
        default="threshold",
        help="evaluation method passed to the upstream script",
    )
    recon_runtime_mainline_parser.add_argument(
        "--similarity-method",
        default="cosine",
        help="similarity method passed to cal_embedding.py",
    )
    recon_runtime_mainline_parser.add_argument(
        "--image-encoder",
        default="deit",
        help="image encoder passed to cal_embedding.py",
    )

    dit_sample_smoke_parser = subparsers.add_parser(
        "run-dit-sample-smoke",
        help="run the official DiT sample script and capture the generated sample image",
    )
    dit_sample_smoke_parser.add_argument(
        "--workspace",
        required=True,
        help="workspace directory for DiT sample smoke artifacts",
    )
    dit_sample_smoke_parser.add_argument(
        "--repo-root",
        default="external/DiT",
        help="path to local official DiT repository root",
    )
    dit_sample_smoke_parser.add_argument(
        "--model",
        default="DiT-XL/2",
        help="DiT model name passed to the official sample script",
    )
    dit_sample_smoke_parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="image size passed to the official sample script",
    )
    dit_sample_smoke_parser.add_argument(
        "--num-sampling-steps",
        type=int,
        default=2,
        help="number of diffusion sampling steps",
    )
    dit_sample_smoke_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed passed to the official sample script",
    )
    dit_sample_smoke_parser.add_argument(
        "--ckpt",
        default=None,
        help="optional local DiT checkpoint path; omitted means the official auto-download path",
    )

    variation_synth_smoke_parser = subparsers.add_parser(
        "run-variation-synth-smoke",
        help="run a synthetic smoke for the API-only variation attack line",
    )
    variation_synth_smoke_parser.add_argument(
        "--workspace",
        required=True,
        help="workspace directory for variation synthetic smoke artifacts",
    )

    blackbox_status_parser = subparsers.add_parser(
        "summarize-blackbox-results",
        help="aggregate black-box experiment summaries into a single machine-readable report",
    )
    blackbox_status_parser.add_argument(
        "--experiments-root",
        required=True,
        help="root directory containing experiment summary subdirectories",
    )
    blackbox_status_parser.add_argument(
        "--workspace",
        required=True,
        help="workspace directory for the aggregated black-box status report",
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

    if args.command == "plan-recon":
        config = load_audit_config(args.config)
        plan = build_recon_plan(config)
        print(json.dumps(asdict(plan), indent=2, ensure_ascii=True))
        return 0

    if args.command == "plan-variation":
        config = load_audit_config(args.config)
        plan = build_variation_plan(config)
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

    if args.command == "probe-recon-assets":
        config = load_audit_config(args.config)
        payload = explain_recon_assets(config)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0 if payload["status"] == "ready" else 1

    if args.command == "probe-recon-score-artifacts":
        payload = probe_recon_score_artifacts(args.artifact_dir)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0 if payload["status"] == "ready" else 1

    if args.command == "probe-recon-runtime-assets":
        payload = probe_recon_runtime_assets(
            target_member_dataset=args.target_member_dataset,
            target_nonmember_dataset=args.target_nonmember_dataset,
            shadow_member_dataset=args.shadow_member_dataset,
            shadow_nonmember_dataset=args.shadow_nonmember_dataset,
            target_model_dir=args.target_model_dir,
            shadow_model_dir=args.shadow_model_dir,
            repo_root=args.repo_root,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0 if payload["status"] == "ready" else 1

    if args.command == "probe-dit-assets":
        payload = probe_dit_assets(
            repo_root=args.repo_root,
            ckpt=args.ckpt,
            model=args.model,
            image_size=args.image_size,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0 if payload["status"] == "ready" else 1

    if args.command == "probe-variation-assets":
        config = load_audit_config(args.config)
        payload = explain_variation_assets(config)
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

    if args.command == "dry-run-recon":
        config = load_audit_config(args.config)
        exit_code, payload = probe_recon_dry_run(config, args.repo_root)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return exit_code

    if args.command == "dry-run-variation":
        config = load_audit_config(args.config)
        exit_code, payload = probe_variation_dry_run(config)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return exit_code

    if args.command == "run-clid-dry-run-smoke":
        payload = run_clid_dry_run_smoke(
            args.workspace,
            repo_root=args.repo_root,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0 if payload["status"] == "ready" else 1

    if args.command == "summarize-clid-artifacts":
        payload = summarize_clid_artifacts(
            artifact_dir=args.artifact_dir,
            workspace=args.workspace,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0

    if args.command == "run-recon-eval-smoke":
        payload = run_recon_eval_smoke(args.workspace)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0

    if args.command == "summarize-recon-artifacts":
        payload = summarize_recon_artifacts(
            artifact_dir=args.artifact_dir,
            workspace=args.workspace,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0

    if args.command == "run-recon-upstream-eval-smoke":
        payload = run_recon_upstream_eval_smoke(
            args.workspace,
            repo_root=args.repo_root,
            method=args.method,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0 if payload["status"] == "ready" else 1

    if args.command == "run-recon-mainline-smoke":
        payload = run_recon_mainline_smoke(
            args.workspace,
            repo_root=args.repo_root,
            method=args.method,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0 if payload["status"] == "ready" else 1

    if args.command == "run-recon-artifact-mainline":
        payload = run_recon_artifact_mainline(
            artifact_dir=args.artifact_dir,
            workspace=args.workspace,
            repo_root=args.repo_root,
            method=args.method,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0 if payload["status"] == "ready" else 1

    if args.command == "run-recon-runtime-mainline":
        payload = run_recon_runtime_mainline(
            target_member_dataset=args.target_member_dataset,
            target_nonmember_dataset=args.target_nonmember_dataset,
            shadow_member_dataset=args.shadow_member_dataset,
            shadow_nonmember_dataset=args.shadow_nonmember_dataset,
            target_model_dir=args.target_model_dir,
            shadow_model_dir=args.shadow_model_dir,
            workspace=args.workspace,
            repo_root=args.repo_root,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            num_validation_images=args.num_validation_images,
            inference_steps=args.inference_steps,
            gpu=args.gpu,
            backend=args.backend,
            scheduler=args.scheduler,
            method=args.method,
            similarity_method=args.similarity_method,
            image_encoder=args.image_encoder,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0 if payload["status"] == "ready" else 1

    if args.command == "run-dit-sample-smoke":
        payload = run_dit_sample_smoke(
            workspace=args.workspace,
            repo_root=args.repo_root,
            model=args.model,
            image_size=args.image_size,
            num_sampling_steps=args.num_sampling_steps,
            seed=args.seed,
            ckpt=args.ckpt,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0 if payload["status"] == "ready" else 1

    if args.command == "run-variation-synth-smoke":
        payload = run_variation_synthetic_smoke(args.workspace)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0

    if args.command == "summarize-blackbox-results":
        payload = build_blackbox_status_report(
            experiments_root=args.experiments_root,
            workspace=args.workspace,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0

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
