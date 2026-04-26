import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image


def _write_png(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size=size, color=color).save(path)


def _make_checkpoint_dir(root: Path, step: int = 9600, with_model: bool = True) -> Path:
    checkpoint_dir = root / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if with_model:
        (checkpoint_dir / "model.safetensors").write_bytes(b"fake-model")
    return checkpoint_dir


class H2AdapterTests(unittest.TestCase):
    def test_probe_h2_assets_reports_ready_for_local_assets(self) -> None:
        from diffaudit.defenses.h2_adapter import probe_h2_assets

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_root = root / "checkpoints" / "target"
            _make_checkpoint_dir(checkpoint_root)

            member_dir = root / "datasets" / "target-member"
            nonmember_dir = root / "datasets" / "target-nonmember"
            _write_png(member_dir / "member-000.png", (32, 32), (255, 0, 0))
            _write_png(member_dir / "member-001.png", (32, 32), (0, 255, 0))
            _write_png(nonmember_dir / "nonmember-000.png", (32, 32), (0, 0, 255))
            _write_png(nonmember_dir / "nonmember-001.png", (32, 32), (255, 255, 0))

            payload = probe_h2_assets(
                checkpoint_root=checkpoint_root,
                member_dataset_dir=member_dir,
                nonmember_dataset_dir=nonmember_dir,
                packet_cap=4,
            )

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["method"], "privacy-aware-adapter")
        self.assertEqual(payload["contract_stage"], "asset-probe")
        self.assertEqual(payload["checkpoint"]["source_kind"], "checkpoint-root-latest")
        self.assertTrue(payload["checks"]["checkpoint_model_file"])
        self.assertEqual(payload["packet"]["member_count"], 2)
        self.assertEqual(payload["packet"]["nonmember_count"], 2)
        self.assertEqual(payload["packet"]["effective_packet_size"], 2)
        self.assertTrue(payload["compatibility"]["layout_compatible"])
        self.assertTrue(payload["compatibility"]["packet_within_cap"])

    def test_probe_h2_assets_blocks_when_nonmember_root_missing(self) -> None:
        from diffaudit.defenses.h2_adapter import probe_h2_assets

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_root = root / "checkpoints" / "target"
            _make_checkpoint_dir(checkpoint_root)
            member_dir = root / "datasets" / "target-member"
            _write_png(member_dir / "member-000.png", (32, 32), (255, 0, 0))

            payload = probe_h2_assets(
                checkpoint_root=checkpoint_root,
                member_dataset_dir=member_dir,
                nonmember_dataset_dir=root / "datasets" / "target-nonmember-missing",
                packet_cap=4,
            )

        self.assertEqual(payload["status"], "blocked")
        self.assertIn("nonmember_dataset_dir", payload["missing_keys"])
        self.assertEqual(payload["blocker_reason"], "missing-required-assets")

    def test_probe_h2_assets_blocks_when_checkpoint_identity_missing(self) -> None:
        from diffaudit.defenses.h2_adapter import probe_h2_assets

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_root = root / "checkpoints" / "target"
            _make_checkpoint_dir(checkpoint_root, with_model=False)

            member_dir = root / "datasets" / "target-member"
            nonmember_dir = root / "datasets" / "target-nonmember"
            _write_png(member_dir / "member-000.png", (32, 32), (255, 0, 0))
            _write_png(nonmember_dir / "nonmember-000.png", (32, 32), (0, 0, 255))

            payload = probe_h2_assets(
                checkpoint_root=checkpoint_root,
                member_dataset_dir=member_dir,
                nonmember_dataset_dir=nonmember_dir,
                packet_cap=4,
            )

        self.assertEqual(payload["status"], "blocked")
        self.assertIn("checkpoint_model_file", payload["missing_keys"])
        self.assertEqual(payload["blocker_reason"], "checkpoint-identity-unresolved")

    def test_cli_probe_h2_assets_reports_layout_mismatch(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_root = root / "checkpoints" / "target"
            _make_checkpoint_dir(checkpoint_root)

            member_dir = root / "datasets" / "target-member"
            nonmember_dir = root / "datasets" / "target-nonmember"
            _write_png(member_dir / "member-000.png", (32, 32), (255, 0, 0))
            _write_png(nonmember_dir / "nonmember-000.png", (64, 64), (0, 0, 255))

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "probe-h2-assets",
                        "--checkpoint-root",
                        str(checkpoint_root),
                        "--member-dataset-dir",
                        str(member_dir),
                        "--nonmember-dataset-dir",
                        str(nonmember_dir),
                        "--packet-cap",
                        "4",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["status"], "blocked")
        self.assertFalse(payload["compatibility"]["layout_compatible"])
        self.assertEqual(payload["compatibility"]["member_image_shape"], [32, 32, 3])
        self.assertEqual(payload["compatibility"]["nonmember_image_shape"], [64, 64, 3])
        self.assertEqual(payload["blocker_reason"], "image-layout-mismatch")

    def test_prepare_h2_contract_writes_manifest_and_summary(self) -> None:
        from diffaudit.defenses.h2_adapter import prepare_h2_contract

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_root = root / "checkpoints" / "target"
            _make_checkpoint_dir(checkpoint_root)

            member_dir = root / "datasets" / "target-member"
            nonmember_dir = root / "datasets" / "target-nonmember"
            _write_png(member_dir / "member-000.png", (32, 32), (255, 0, 0))
            _write_png(nonmember_dir / "nonmember-000.png", (32, 32), (0, 0, 255))

            workspace = root / "runs" / "h2-prepare"
            payload = prepare_h2_contract(
                workspace=workspace,
                checkpoint_root=checkpoint_root,
                member_dataset_dir=member_dir,
                nonmember_dataset_dir=nonmember_dir,
                packet_cap=4,
                rank=8,
                lambda_coeff=0.25,
                num_epochs=3,
                batch_size=2,
            )

            summary_path = workspace / "summary.json"
            manifest_path = workspace / "manifest.json"
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["contract_stage"], "prepare-contract")
            self.assertTrue(summary_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertEqual(payload["runtime"]["rank"], 8)
            self.assertEqual(payload["runtime"]["lambda_coeff"], 0.25)
            self.assertEqual(payload["runtime"]["num_epochs"], 3)
            self.assertEqual(payload["runtime"]["batch_size"], 2)
            self.assertEqual(payload["packet"]["effective_packet_size"], 1)

    def test_cli_prepare_h2_contract_blocks_when_probe_is_not_ready(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_root = root / "checkpoints" / "target"
            _make_checkpoint_dir(checkpoint_root)
            member_dir = root / "datasets" / "target-member"
            _write_png(member_dir / "member-000.png", (32, 32), (255, 0, 0))

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "prepare-h2-contract",
                        "--workspace",
                        str(root / "runs" / "h2-prepare-blocked"),
                        "--checkpoint-root",
                        str(checkpoint_root),
                        "--member-dataset-dir",
                        str(member_dir),
                        "--nonmember-dataset-dir",
                        str(root / "datasets" / "missing-nonmember"),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["blocker_reason"], "probe-not-ready")
        self.assertIn("probe", payload)

    def test_run_h2_defense_pilot_writes_summary_and_training_artifacts(self) -> None:
        from diffaudit.defenses.h2_adapter import prepare_h2_contract, run_h2_defense_pilot

        run_workspace = None

        def fake_run(command, cwd):
            del command, cwd
            output_dir = run_workspace / "run-output"
            final_dir = output_dir / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "config.json").write_text(
                json.dumps({"device": "cpu", "num_epochs": 3}, indent=2),
                encoding="utf-8",
            )
            (final_dir / "checkpoint_meta.json").write_text(
                json.dumps({"method": "smp", "lambda": 0.25}, indent=2),
                encoding="utf-8",
            )
            (final_dir / "lora_summary.json").write_text(
                json.dumps({"num_lora_layers": 12, "total_lora_params": 49152}, indent=2),
                encoding="utf-8",
            )
            (final_dir / "training_log.json").write_text(
                json.dumps(
                    [
                        {
                            "step": 0,
                            "adaptation_loss": 1.2,
                            "mi_gain": 0.5,
                            "proxy_loss": 0.7,
                            "objective": 1.26,
                        }
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
            (final_dir / "lora_weights.pt").write_bytes(b"lora")
            (final_dir / "proxy_weights.pt").write_bytes(b"proxy")
            return 0, "ok", ""

        def fake_materialize(**kwargs):
            output_dir = Path(kwargs["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            target = output_dir / "model.safetensors"
            target.write_bytes(b"merged")
            return target

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_root = root / "checkpoints" / "target"
            _make_checkpoint_dir(checkpoint_root)
            member_dir = root / "datasets" / "target-member"
            nonmember_dir = root / "datasets" / "target-nonmember"
            _write_png(member_dir / "member-000.png", (32, 32), (255, 0, 0))
            _write_png(member_dir / "member-001.png", (32, 32), (0, 255, 0))
            _write_png(nonmember_dir / "nonmember-000.png", (32, 32), (0, 0, 255))
            _write_png(nonmember_dir / "nonmember-001.png", (32, 32), (255, 255, 0))

            prepared_workspace = root / "prepare"
            prepare_h2_contract(
                workspace=prepared_workspace,
                checkpoint_root=checkpoint_root,
                member_dataset_dir=member_dir,
                nonmember_dataset_dir=nonmember_dir,
                packet_cap=4,
                rank=8,
                lambda_coeff=0.25,
                num_epochs=3,
                batch_size=2,
            )
            manifest_path = prepared_workspace / "manifest.json"
            run_workspace = root / "run"

            with patch("diffaudit.defenses.h2_adapter._run_h2_training_command", side_effect=fake_run), patch(
                "diffaudit.defenses.h2_adapter._materialize_h2_review_checkpoint",
                side_effect=fake_materialize,
            ):
                payload = run_h2_defense_pilot(
                    workspace=run_workspace,
                    manifest_path=manifest_path,
                    member_limit=1,
                    nonmember_limit=1,
                    seed=7,
                )

            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["contract_stage"], "run-pilot")
            self.assertEqual(payload["executed_packet"]["member_count"], 1)
            self.assertEqual(payload["executed_packet"]["nonmember_count"], 1)
            self.assertEqual(payload["runtime"]["seed"], 7)
            self.assertTrue(Path(payload["artifact_paths"]["training_log"]).exists())
            self.assertTrue(Path(payload["artifact_paths"]["checkpoint_meta"]).exists())
            self.assertTrue(Path(payload["artifact_paths"]["review_checkpoint"]).exists())
            self.assertEqual(payload["metrics"]["step0_objective"], 1.26)

    def test_cli_run_h2_defense_pilot_blocks_when_manifest_missing(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "run-h2-defense-pilot",
                        "--workspace",
                        str(root / "run"),
                        "--manifest",
                        str(root / "missing-manifest.json"),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["blocker_reason"], "prepare-manifest-missing")

    def test_review_h2_defense_pilot_compares_baseline_and_defended(self) -> None:
        from diffaudit.defenses.h2_adapter import review_h2_defense_pilot

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shadow_summary = root / "shadow-summary.json"
            shadow_summary.write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "mode": "loss-score-export",
                        "artifact_paths": {"shadow_specs": [{"name": "shadow-01"}]},
                        "exports": {
                            "shadow_01_member": {"output_path": str(root / "shadow_01_member.pt")},
                            "shadow_01_non_member": {"output_path": str(root / "shadow_01_non_member.pt")},
                        },
                        "runtime": {
                            "resolution": 32,
                            "ddpm_num_steps": 20,
                            "sampling_frequency": 2,
                            "attack_method": 1,
                            "prediction_type": "epsilon",
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            run_summary = root / "run-summary.json"
            packet_member_dir = root / "packet" / "member"
            packet_nonmember_dir = root / "packet" / "nonmember"
            packet_member_dir.mkdir(parents=True, exist_ok=True)
            packet_nonmember_dir.mkdir(parents=True, exist_ok=True)
            _write_png(packet_member_dir / "0000-member.png", (32, 32), (255, 0, 0))
            _write_png(packet_nonmember_dir / "0000-nonmember.png", (32, 32), (0, 0, 255))
            run_summary.write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "mode": "run-pilot",
                        "baseline_checkpoint": {
                            "checkpoint_root": str(root / "baseline-root"),
                            "resolved_checkpoint_dir": str(root / "baseline-root" / "checkpoint-1"),
                        },
                        "defended_checkpoint": str(root / "defended" / "final"),
                        "executed_packet": {
                            "member_stage_dir": str(packet_member_dir),
                            "nonmember_stage_dir": str(packet_nonmember_dir),
                            "member_count": 1,
                            "nonmember_count": 1,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            received_checkpoint_dirs: list[str] = []

            def fake_extract(**kwargs):
                output_path = Path(kwargs["output_path"])
                records_path = Path(kwargs["records_path"])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                records_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"tensor")
                records_path.write_text("{}", encoding="utf-8")
                received_checkpoint_dirs.append(str(kwargs["checkpoint_dir"]))
                return {
                    "status": "ready",
                    "output_path": str(output_path),
                    "records_path": str(records_path),
                    "checkpoint_dir": str(kwargs["checkpoint_dir"]),
                    "sample_count": 1,
                    "score_stats": {"mean": 0.1, "min": 0.1, "max": 0.1},
                }

            def fake_eval(*, workspace, packet_summary, evaluation_style="threshold-transfer", provenance_status="workspace-verified"):
                del workspace, evaluation_style, provenance_status
                packet_name = Path(packet_summary).name
                auc = 0.61 if "baseline" in packet_name else 0.57
                return {
                    "status": "ready",
                    "target_transfer": {
                        "metrics": {
                            "auc": auc,
                            "asr": 0.55,
                            "tpr_at_1pct_fpr": 0.1,
                            "tpr_at_0_1pct_fpr": 0.02,
                        }
                    },
                    "target_self_diagnostic": {
                        "metrics": {
                            "auc": auc,
                            "asr": 0.56,
                            "tpr_at_1pct_fpr": 0.11,
                            "tpr_at_0_1pct_fpr": 0.03,
                        }
                    },
                    "shadow_pool": {"metrics": {"auc": 0.5}},
                }

            with patch("diffaudit.attacks.gsa._extract_gsa_loss_scores", side_effect=fake_extract), patch(
                "diffaudit.attacks.gsa.evaluate_gsa_loss_score_packet",
                side_effect=fake_eval,
            ):
                summary = review_h2_defense_pilot(
                    workspace=root / "review",
                    run_summary_path=run_summary,
                    shadow_reference_summary=shadow_summary,
                    noise_seed=9,
                )

        self.assertEqual(summary["status"], "ready")
        self.assertEqual(summary["attacker_mode"], "transfer-only-shadow-threshold")
        self.assertEqual(summary["comparison"]["metric_deltas"]["auc"], -0.04)
        self.assertEqual(summary["comparison"]["metric_deltas"]["tpr_at_0_1pct_fpr"], 0.0)
        self.assertIn(str(root / "defended" / "final"), received_checkpoint_dirs)

    def test_cli_review_h2_defense_pilot_blocks_when_run_summary_missing(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "review-h2-defense-pilot",
                        "--workspace",
                        str(root / "review"),
                        "--run-summary",
                        str(root / "missing-run-summary.json"),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["blocker_reason"], "run-summary-missing")


if __name__ == "__main__":
    unittest.main()
