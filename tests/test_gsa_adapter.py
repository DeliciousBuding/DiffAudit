import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import torch


def create_minimal_gsa_repo(repo_root: Path) -> None:
    ddpm_root = repo_root / "DDPM"
    ddpm_root.mkdir(parents=True)
    (ddpm_root / "gen_l2_gradients_DDPM.py").write_text("# gradient entrypoint\n", encoding="utf-8")
    (ddpm_root / "train_unconditional.py").write_text("# train entrypoint\n", encoding="utf-8")
    (repo_root / "test_attack_accuracy.py").write_text("# attack entrypoint\n", encoding="utf-8")


def create_minimal_gsa_assets(assets_root: Path) -> None:
    for relative in (
        "datasets/target-member",
        "datasets/target-nonmember",
        "datasets/shadow-member",
        "datasets/shadow-nonmember",
        "checkpoints/target/checkpoint-10",
        "checkpoints/shadow/checkpoint-20",
        "manifests",
        "sources",
    ):
        (assets_root / relative).mkdir(parents=True, exist_ok=True)
    (assets_root / "datasets" / "target-member" / "sample-a.png").write_bytes(b"png")
    (assets_root / "datasets" / "target-nonmember" / "sample-b.png").write_bytes(b"png")
    (assets_root / "datasets" / "shadow-member" / "sample-c.png").write_bytes(b"png")
    (assets_root / "datasets" / "shadow-nonmember" / "sample-d.png").write_bytes(b"png")
    (assets_root / "manifests" / "split-manifest.json").write_text("{}", encoding="utf-8")
    (assets_root / "sources" / "cifar-10-python.tar.gz").write_bytes(b"archive")


class GsaAdapterTests(unittest.TestCase):
    def test_cli_probes_gsa_assets(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "GSA"
            assets_root = root / "assets"
            create_minimal_gsa_repo(repo_root)
            create_minimal_gsa_assets(assets_root)

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "probe-gsa-assets",
                        "--repo-root",
                        str(repo_root),
                        "--assets-root",
                        str(assets_root),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["checks"]["target_checkpoint"])
        self.assertTrue(payload["checks"]["shadow_checkpoints"])
        self.assertTrue(payload["checks"]["manifest_file"])

    def test_cli_runs_gsa_runtime_mainline(self) -> None:
        from diffaudit.cli import main

        def fake_subprocess_run(command, cwd=None, capture_output=None, text=None, check=None, env=None):
            del cwd, capture_output, text, check, env
            if "--output_name" in command:
                output_path = Path(command[command.index("--output_name") + 1])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                base = 0.9 if "member" in output_path.name else 0.1
                tensor = torch.tensor(
                    [
                        [base, base + 0.1, base + 0.2],
                        [base + 0.05, base + 0.15, base + 0.25],
                    ],
                    dtype=torch.float32,
                )
                torch.save(tensor, output_path)
                return type(
                    "Completed",
                    (),
                    {"returncode": 0, "stdout": "gradient ok\n", "stderr": ""},
                )()
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "attack ok\n", "stderr": ""},
            )()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "GSA"
            assets_root = root / "assets"
            create_minimal_gsa_repo(repo_root)
            create_minimal_gsa_assets(assets_root)

            stdout = StringIO()
            with patch("diffaudit.attacks.gsa.subprocess.run", side_effect=fake_subprocess_run):
                with redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "run-gsa-runtime-mainline",
                            "--workspace",
                            str(root / "gsa-runtime-mainline"),
                            "--repo-root",
                            str(repo_root),
                            "--assets-root",
                            str(assets_root),
                            "--resolution",
                            "32",
                            "--ddpm-num-steps",
                            "20",
                            "--sampling-frequency",
                            "2",
                            "--attack-method",
                            "1",
                        ]
                    )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["mode"], "runtime-mainline")
            self.assertEqual(payload["workspace_name"], "gsa-runtime-mainline")
            self.assertEqual(payload["evidence_level"], "runtime-mainline")
            self.assertEqual(payload["asset_grade"], "real-asset-closed-loop")
            self.assertEqual(payload["contract_stage"], "target")
            self.assertIn("auc", payload["metrics"])
            self.assertIn("asr", payload["metrics"])
            self.assertIn("tpr_at_1pct_fpr", payload["metrics"])
            self.assertIn("tpr_at_0_1pct_fpr", payload["metrics"])
            self.assertTrue(Path(payload["artifact_paths"]["summary"]).exists())
            self.assertTrue(Path(payload["artifact_paths"]["target_member_gradients"]).exists())

    def test_cli_runs_gsa_runtime_mainline_with_bounded_max_samples(self) -> None:
        from diffaudit.cli import main

        def fake_subprocess_run(command, cwd=None, capture_output=None, text=None, check=None, env=None):
            del cwd, capture_output, text, check, env
            if "--output_name" in command:
                output_path = Path(command[command.index("--output_name") + 1])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                base = 0.9 if "member" in output_path.name else 0.1
                tensor = torch.tensor(
                    [
                        [base, base + 0.1, base + 0.2],
                        [base + 0.05, base + 0.15, base + 0.25],
                        [base + 0.1, base + 0.2, base + 0.3],
                        [base + 0.15, base + 0.25, base + 0.35],
                    ],
                    dtype=torch.float32,
                )
                torch.save(tensor, output_path)
                return type(
                    "Completed",
                    (),
                    {"returncode": 0, "stdout": "gradient ok\n", "stderr": ""},
                )()
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "attack ok\n", "stderr": ""},
            )()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "GSA"
            assets_root = root / "assets"
            create_minimal_gsa_repo(repo_root)
            create_minimal_gsa_assets(assets_root)

            stdout = StringIO()
            with patch("diffaudit.attacks.gsa.subprocess.run", side_effect=fake_subprocess_run):
                with redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "run-gsa-runtime-mainline",
                            "--workspace",
                            str(root / "gsa-runtime-mainline-bounded"),
                            "--repo-root",
                            str(repo_root),
                            "--assets-root",
                            str(assets_root),
                            "--resolution",
                            "32",
                            "--ddpm-num-steps",
                            "20",
                            "--sampling-frequency",
                            "2",
                            "--attack-method",
                            "1",
                            "--max-samples",
                            "2",
                        ]
                    )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["runtime"]["max_samples"], 2)
            self.assertEqual(payload["metrics"]["target_eval_size"], 4)
            self.assertLessEqual(payload["metrics"]["shadow_train_size"], 4)

    def test_cli_runs_gsa_runtime_intervention_review_with_frozen_mask_summary(self) -> None:
        from diffaudit.cli import main

        received_extraction_caps: list[int | None] = []

        def fake_extract(
            *,
            output_path,
            channel_indices=None,
            extraction_max_samples=None,
            **kwargs,
        ):
            del kwargs
            received_extraction_caps.append(extraction_max_samples)
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            base = 0.9 if "member" in output_path.name else 0.1
            if channel_indices:
                base = base - 0.2 if "member" in output_path.name else base + 0.2
            tensor = torch.tensor(
                [
                    [base, base + 0.1, base + 0.2],
                    [base + 0.05, base + 0.15, base + 0.25],
                    [base + 0.1, base + 0.2, base + 0.3],
                    [base + 0.15, base + 0.25, base + 0.35],
                ],
                dtype=torch.float32,
            )
            torch.save(tensor, output_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "GSA"
            assets_root = root / "assets"
            create_minimal_gsa_repo(repo_root)
            create_minimal_gsa_assets(assets_root)
            mask_summary = root / "mask-summary.json"
            mask_summary.write_text(
                json.dumps(
                    {
                        "mode": "inmodel-packet-export",
                        "requested": {
                            "layer_selector": "mid_block.attentions.0.to_v",
                            "checkpoint_dir": str(
                                assets_root / "checkpoints" / "target" / "checkpoint-10"
                            ),
                            "sample_id": "target-member/sample-a.png",
                            "control_sample_id": "target-nonmember/sample-b.png",
                            "timestep": 999,
                            "mask_seed": 11,
                            "noise_seed": 7,
                        },
                        "mask": {
                            "mask_kind": "top_abs_delta_k",
                            "channel_indices": [0, 2],
                            "k": 2,
                            "alpha": 0.5,
                        },
                        "metrics": {
                            "selected_delta_retention_ratio": 0.5,
                            "off_mask_drift": 0.0,
                            "epsilon_prediction_rms_drift_mean": 1e-6,
                            "epsilon_prediction_max_abs_drift_mean": 2e-6,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            stdout = StringIO()
            with patch(
                "diffaudit.attacks.gsa._extract_gsa_gradients_with_fixed_mask",
                side_effect=fake_extract,
                create=True,
            ), patch(
                "diffaudit.attacks.gsa._evaluate_gsa_closed_loop",
                side_effect=[
                    {
                        "auc": 0.99,
                        "asr": 0.95,
                        "tpr_at_1pct_fpr": 0.6,
                        "tpr_at_0_1pct_fpr": 0.2,
                        "shadow_train_size": 8,
                        "target_eval_size": 4,
                    },
                    {
                        "auc": 0.88,
                        "asr": 0.8,
                        "tpr_at_1pct_fpr": 0.25,
                        "tpr_at_0_1pct_fpr": 0.0,
                        "shadow_train_size": 8,
                        "target_eval_size": 4,
                    },
                ],
                create=True,
            ):
                with redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "run-gsa-runtime-intervention-review",
                            "--workspace",
                            str(root / "gsa-runtime-intervention-review"),
                            "--repo-root",
                            str(repo_root),
                            "--assets-root",
                            str(assets_root),
                            "--mask-summary",
                            str(mask_summary),
                            "--max-samples",
                            "2",
                        ]
                    )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["mode"], "runtime-intervention-review")
            self.assertEqual(payload["runtime"]["max_samples"], 2)
            self.assertEqual(payload["runtime"]["extraction_max_samples"], 2)
            self.assertEqual(payload["mask"]["channel_indices"], [0, 2])
            self.assertEqual(payload["baseline"]["metrics"]["auc"], 0.99)
            self.assertEqual(payload["intervened"]["metrics"]["auc"], 0.88)
            self.assertEqual(payload["metric_deltas"]["auc_delta"], -0.11)
            self.assertTrue(received_extraction_caps)
            self.assertTrue(all(cap == 2 for cap in received_extraction_caps))
            self.assertTrue(Path(payload["artifact_paths"]["summary"]).exists())

    def test_cli_exports_gsa_loss_score_packet(self) -> None:
        from diffaudit.cli import main

        def fake_export(
            *,
            output_path,
            records_path,
            extraction_max_samples=None,
            **kwargs,
        ):
            del kwargs
            output_path = Path(output_path)
            records_path = Path(records_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            records_path.parent.mkdir(parents=True, exist_ok=True)
            base = 0.9 if "member" in output_path.name else 0.1
            tensor = torch.tensor([base, base + 0.05], dtype=torch.float32)
            torch.save(tensor, output_path)
            records_path.write_text(
                json.dumps({"sample_index": 0, "sample_path": "sample-a.png", "loss_score": float(base)}),
                encoding="utf-8",
            )
            return {
                "status": "ready",
                "output_path": str(output_path),
                "records_path": str(records_path),
                "checkpoint_dir": "checkpoint-10",
                "sample_count": 1 if extraction_max_samples == 1 else 2,
                "score_stats": {
                    "mean": round(float(tensor.mean().item()), 6),
                    "min": round(float(tensor.min().item()), 6),
                    "max": round(float(tensor.max().item()), 6),
                },
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "GSA"
            assets_root = root / "assets"
            create_minimal_gsa_repo(repo_root)
            create_minimal_gsa_assets(assets_root)

            stdout = StringIO()
            with patch(
                "diffaudit.attacks.gsa._extract_gsa_loss_scores",
                side_effect=fake_export,
                create=True,
            ):
                with redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "export-gsa-loss-score-packet",
                            "--workspace",
                            str(root / "gsa-loss-score-packet"),
                            "--repo-root",
                            str(repo_root),
                            "--assets-root",
                            str(assets_root),
                            "--extraction-max-samples",
                            "1",
                        ]
                    )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["mode"], "loss-score-export")
            self.assertEqual(payload["runtime"]["extraction_max_samples"], 1)
            self.assertTrue(Path(payload["artifact_paths"]["summary"]).exists())
            self.assertTrue(Path(payload["artifact_paths"]["target_member_scores"]).exists())

    def test_cli_evaluates_gsa_loss_score_packet_with_shadow_only_orientation(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            export_root = root / "gsa-loss-score-export"
            scores_root = export_root / "loss-scores"
            scores_root.mkdir(parents=True, exist_ok=True)

            artifacts = {
                "target_member": scores_root / "target_member-loss-scores.pt",
                "target_non_member": scores_root / "target_non_member-loss-scores.pt",
                "shadow_01_member": scores_root / "shadow_01_member-loss-scores.pt",
                "shadow_01_non_member": scores_root / "shadow_01_non_member-loss-scores.pt",
            }
            torch.save(torch.tensor([0.6, 0.7], dtype=torch.float32), artifacts["target_member"])
            torch.save(torch.tensor([0.3, 0.4], dtype=torch.float32), artifacts["target_non_member"])
            torch.save(torch.tensor([0.2, 0.3], dtype=torch.float32), artifacts["shadow_01_member"])
            torch.save(torch.tensor([0.7, 0.8], dtype=torch.float32), artifacts["shadow_01_non_member"])

            packet_summary = export_root / "summary.json"
            packet_summary.write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "mode": "loss-score-export",
                        "asset_grade": "real-asset-closed-loop",
                        "runtime": {
                            "extraction_max_samples": 64,
                            "attack_method": 1,
                        },
                        "artifact_paths": {
                            "shadow_specs": [
                                {
                                    "name": "shadow-01",
                                }
                            ]
                        },
                        "exports": {
                            key: {
                                "status": "ready",
                                "output_path": str(path),
                            }
                            for key, path in artifacts.items()
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "evaluate-gsa-loss-score-packet",
                        "--workspace",
                        str(root / "gsa-loss-score-threshold-eval"),
                        "--packet-summary",
                        str(packet_summary),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["mode"], "loss-score-threshold-eval")
            self.assertEqual(payload["shadow_pool"]["metrics"]["score_direction"], "member-lower")
            self.assertEqual(payload["target_transfer"]["metrics"]["score_direction"], "member-lower")
            self.assertEqual(payload["target_transfer"]["metrics"]["inferred_score_direction"], "member-higher")
            self.assertEqual(payload["target_self_diagnostic"]["metrics"]["score_direction"], "member-higher")
            self.assertTrue(Path(payload["artifact_paths"]["summary"]).exists())

    def test_cli_evaluates_gsa_loss_score_packet_with_gaussian_lr_transfer(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            export_root = root / "gsa-loss-score-export"
            scores_root = export_root / "loss-scores"
            scores_root.mkdir(parents=True, exist_ok=True)

            artifacts = {
                "target_member": scores_root / "target_member-loss-scores.pt",
                "target_non_member": scores_root / "target_non_member-loss-scores.pt",
                "shadow_01_member": scores_root / "shadow_01_member-loss-scores.pt",
                "shadow_01_non_member": scores_root / "shadow_01_non_member-loss-scores.pt",
            }
            torch.save(torch.tensor([0.58, 0.61], dtype=torch.float32), artifacts["target_member"])
            torch.save(torch.tensor([0.31, 0.35], dtype=torch.float32), artifacts["target_non_member"])
            torch.save(torch.tensor([0.62, 0.64], dtype=torch.float32), artifacts["shadow_01_member"])
            torch.save(torch.tensor([0.28, 0.32], dtype=torch.float32), artifacts["shadow_01_non_member"])

            packet_summary = export_root / "summary.json"
            packet_summary.write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "mode": "loss-score-export",
                        "asset_grade": "real-asset-closed-loop",
                        "runtime": {
                            "extraction_max_samples": 64,
                            "attack_method": 1,
                        },
                        "artifact_paths": {
                            "shadow_specs": [
                                {
                                    "name": "shadow-01",
                                }
                            ]
                        },
                        "exports": {
                            key: {
                                "status": "ready",
                                "output_path": str(path),
                            }
                            for key, path in artifacts.items()
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "evaluate-gsa-loss-score-packet",
                        "--workspace",
                        str(root / "gsa-loss-score-lr-eval"),
                        "--packet-summary",
                        str(packet_summary),
                        "--evaluation-style",
                        "gaussian-likelihood-ratio-transfer",
                    ]
                )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["mode"], "loss-score-lr-eval")
            self.assertEqual(payload["runtime"]["evaluation_style"], "gaussian-likelihood-ratio-transfer")
            self.assertEqual(payload["runtime"]["orientation_source"], "pooled-shadow-gaussian-lr")
            self.assertIn("density_fit", payload["shadow_pool"])
            self.assertGreater(payload["target_transfer"]["metrics"]["auc"], 0.5)
            self.assertTrue(Path(payload["artifact_paths"]["summary"]).exists())


if __name__ == "__main__":
    unittest.main()
