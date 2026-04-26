import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (32, 32), color=color)
    image.save(path)


class RiskTargetedUnlearningPrepTests(unittest.TestCase):
    def test_prep_prefers_top_fraction_intersection_when_it_is_large_enough(self) -> None:
        from diffaudit.defenses.risk_targeted_unlearning import run_risk_targeted_unlearning_prep

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pia_path = root / "pia.json"
            gsa_path = root / "gsa.json"

            _write_json(
                pia_path,
                {
                    "member_scores": [0.91, 0.88, 0.62, 0.58],
                    "nonmember_scores": [0.41, 0.37, 0.31, 0.25],
                    "member_indices": [10, 11, 12, 13],
                    "nonmember_indices": [20, 21, 22, 23],
                },
            )
            _write_json(
                gsa_path,
                {
                    "member_scores": [0.85, 0.81, 0.57, 0.52],
                    "nonmember_scores": [0.35, 0.32, 0.29, 0.22],
                    "member_indices": [10, 11, 12, 13],
                    "nonmember_indices": [20, 21, 22, 23],
                },
            )

            summary = run_risk_targeted_unlearning_prep(
                workspace=root / "run",
                surface_a_path=pia_path,
                surface_b_path=gsa_path,
                surface_a_name="pia",
                surface_b_name="gsa",
                top_fraction=0.5,
                top_k_values=[2],
            )

        ladder = summary["forget_ladders"]["k2"]
        self.assertEqual(ladder["selection_mode"], "top-fraction-intersection")
        self.assertEqual(ladder["member_indices"], [10, 11])
        self.assertEqual(len(ladder["matched_nonmember_indices"]), 2)
        self.assertEqual(sorted(ladder["matched_nonmember_indices"]), [20, 21])

    def test_prep_falls_back_to_aggregate_percentile_when_intersection_is_too_small(self) -> None:
        from diffaudit.defenses.risk_targeted_unlearning import run_risk_targeted_unlearning_prep

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pia_path = root / "pia.json"
            gsa_path = root / "gsa.json"

            _write_json(
                pia_path,
                {
                    "member_scores": [0.95, 0.88, 0.79, 0.66, 0.51],
                    "nonmember_scores": [0.46, 0.38, 0.33, 0.27, 0.21],
                    "member_indices": [10, 11, 12, 13, 14],
                    "nonmember_indices": [20, 21, 22, 23, 24],
                },
            )
            _write_json(
                gsa_path,
                {
                    "member_scores": [0.92, 0.73, 0.64, 0.82, 0.5],
                    "nonmember_scores": [0.44, 0.31, 0.24, 0.36, 0.19],
                    "member_indices": [10, 11, 12, 13, 14],
                    "nonmember_indices": [20, 21, 22, 23, 24],
                },
            )

            summary = run_risk_targeted_unlearning_prep(
                workspace=root / "run",
                surface_a_path=pia_path,
                surface_b_path=gsa_path,
                surface_a_name="pia",
                surface_b_name="gsa",
                top_fraction=0.4,
                top_k_values=[2],
            )

        ladder = summary["forget_ladders"]["k2"]
        self.assertEqual(ladder["selection_mode"], "aggregate-percentile")
        self.assertEqual(ladder["member_indices"], [10, 11])

    def test_cli_exports_forget_lists_and_machine_readable_summary(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pia_path = root / "pia.json"
            gsa_path = root / "gsa.json"

            _write_json(
                pia_path,
                {
                    "member_scores": [0.91, 0.88, 0.62, 0.58],
                    "nonmember_scores": [0.41, 0.37, 0.31, 0.25],
                    "member_indices": [10, 11, 12, 13],
                    "nonmember_indices": [20, 21, 22, 23],
                },
            )
            _write_json(
                gsa_path,
                {
                    "member_scores": [0.85, 0.81, 0.57, 0.52],
                    "nonmember_scores": [0.35, 0.32, 0.29, 0.22],
                    "member_indices": [10, 11, 12, 13],
                    "nonmember_indices": [20, 21, 22, 23],
                },
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "prepare-risk-targeted-unlearning-pilot",
                        "--workspace",
                        str(root / "run"),
                        "--surface-a",
                        str(pia_path),
                        "--surface-b",
                        str(gsa_path),
                        "--surface-a-name",
                        "pia",
                        "--surface-b-name",
                        "gsa",
                        "--top-fraction",
                        "0.5",
                        "--top-k",
                        "2",
                    ]
                )

            payload = json.loads(stdout.getvalue())
            summary_exists = Path(payload["artifact_paths"]["summary"]).exists()
            forget_file = Path(payload["artifact_paths"]["forget_member_index_files"]["k2"])
            forget_file_exists = forget_file.exists()
            forget_lines = forget_file.read_text(encoding="utf-8").splitlines()

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertIn("k2", payload["forget_ladders"])
        self.assertTrue(summary_exists)
        self.assertTrue(forget_file_exists)
        self.assertEqual(forget_lines, ["10", "11"])


class RiskTargetedUnlearningPilotTests(unittest.TestCase):
    def test_prepare_member_subsets_resolves_forget_and_retain_paths(self) -> None:
        from diffaudit.defenses.risk_targeted_unlearning import prepare_member_subsets

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            member_dir = root / "target-member"
            member_dir.mkdir(parents=True, exist_ok=True)
            _write_png(member_dir / "00-data_batch_1-00010.png", (255, 0, 0))
            _write_png(member_dir / "00-data_batch_1-00011.png", (0, 255, 0))
            _write_png(member_dir / "00-data_batch_1-00012.png", (0, 0, 255))

            forget_index_file = root / "forget-members-k1.txt"
            forget_index_file.write_text("10\n", encoding="utf-8")

            subsets = prepare_member_subsets(
                member_dataset_dir=member_dir,
                forget_member_index_file=forget_index_file,
                retain_max_samples=1,
            )

        self.assertEqual(subsets["forget_member_indices"], [10])
        self.assertEqual(subsets["retain_member_indices"], [11])
        self.assertEqual(Path(subsets["forget_member_paths"][0]).name, "00-data_batch_1-00010.png")
        self.assertEqual(Path(subsets["retain_member_paths"][0]).name, "00-data_batch_1-00011.png")

    def test_prepare_member_subsets_keeps_all_duplicate_paths_for_one_forget_id(self) -> None:
        from diffaudit.defenses.risk_targeted_unlearning import prepare_member_subsets

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            member_dir = root / "target-member"
            member_dir.mkdir(parents=True, exist_ok=True)
            _write_png(member_dir / "00-data_batch_1-00010.png", (255, 0, 0))
            _write_png(member_dir / "01-data_batch_1-00010.png", (200, 0, 0))
            _write_png(member_dir / "00-data_batch_1-00011.png", (0, 255, 0))

            forget_index_file = root / "forget-members-k1.txt"
            forget_index_file.write_text("10\n", encoding="utf-8")

            subsets = prepare_member_subsets(
                member_dataset_dir=member_dir,
                forget_member_index_file=forget_index_file,
            )

        self.assertEqual(subsets["forget_member_indices"], [10])
        self.assertEqual(len(subsets["forget_member_paths"]), 2)
        self.assertEqual(len(subsets["retain_member_paths"]), 1)

    def test_run_pilot_random_init_single_step_writes_summary_and_checkpoint(self) -> None:
        from diffaudit.defenses.risk_targeted_unlearning import run_risk_targeted_unlearning_pilot

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            member_dir = root / "target-member"
            member_dir.mkdir(parents=True, exist_ok=True)
            _write_png(member_dir / "00-data_batch_1-00010.png", (255, 0, 0))
            _write_png(member_dir / "00-data_batch_1-00011.png", (0, 255, 0))

            forget_index_file = root / "forget-members-k1.txt"
            forget_index_file.write_text("10\n", encoding="utf-8")

            summary = run_risk_targeted_unlearning_pilot(
                workspace=root / "run",
                member_dataset_dir=member_dir,
                forget_member_index_file=forget_index_file,
                random_init=True,
                num_steps=1,
                batch_size=1,
                retain_max_samples=1,
                num_workers=0,
                device="cpu",
                seed=7,
            )

            summary_exists = Path(summary["artifact_paths"]["summary"]).exists()
            checkpoint_exists = Path(summary["artifact_paths"]["checkpoint_dir"], "model.safetensors").exists()

        self.assertEqual(summary["status"], "ready")
        self.assertEqual(summary["runtime"]["executed_steps"], 1)
        self.assertEqual(summary["data"]["forget_member_count"], 1)
        self.assertEqual(summary["data"]["retain_member_count"], 1)
        self.assertTrue(summary_exists)
        self.assertTrue(checkpoint_exists)


class RiskTargetedUnlearningReviewTests(unittest.TestCase):
    def test_export_retained_highrisk_subset_excludes_forget_and_used_nonmembers(self) -> None:
        from diffaudit.defenses.risk_targeted_unlearning import export_retained_highrisk_companion_subset

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            member_records = root / "member-risk-records.jsonl"
            nonmember_records = root / "nonmember-risk-records.jsonl"
            forget_index_file = root / "forget-members-k2.txt"
            matched_nonmember_index_file = root / "matched-nonmembers-k2.txt"

            member_records.write_text(
                "\n".join(
                    [
                        json.dumps({"split_index": 10, "combined_rank": 1, "combined_risk": 0.99}),
                        json.dumps({"split_index": 11, "combined_rank": 2, "combined_risk": 0.95}),
                        json.dumps({"split_index": 12, "combined_rank": 3, "combined_risk": 0.90}),
                        json.dumps({"split_index": 13, "combined_rank": 4, "combined_risk": 0.88}),
                        json.dumps({"split_index": 14, "combined_rank": 5, "combined_risk": 0.60}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            nonmember_records.write_text(
                "\n".join(
                    [
                        json.dumps({"split_index": 20, "combined_rank": 1, "combined_risk": 0.94, "surface_a_percentile": 0.9, "surface_b_percentile": 0.9}),
                        json.dumps({"split_index": 21, "combined_rank": 2, "combined_risk": 0.91, "surface_a_percentile": 0.85, "surface_b_percentile": 0.85}),
                        json.dumps({"split_index": 22, "combined_rank": 3, "combined_risk": 0.89, "surface_a_percentile": 0.8, "surface_b_percentile": 0.8}),
                        json.dumps({"split_index": 23, "combined_rank": 4, "combined_risk": 0.86, "surface_a_percentile": 0.75, "surface_b_percentile": 0.75}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            forget_index_file.write_text("10\n11\n", encoding="utf-8")
            matched_nonmember_index_file.write_text("20\n21\n", encoding="utf-8")

            payload = export_retained_highrisk_companion_subset(
                workspace=root / "retained-subset",
                member_risk_records_path=member_records,
                nonmember_risk_records_path=nonmember_records,
                forget_member_index_file=forget_index_file,
                used_nonmember_index_file=matched_nonmember_index_file,
                subset_size=2,
            )

        self.assertEqual(payload["member_indices"], [12, 13])
        self.assertEqual(payload["matched_nonmember_indices"], [22, 23])

    def test_review_uses_union_allowlist_and_compares_baseline_vs_defended(self) -> None:
        from diffaudit.defenses.risk_targeted_unlearning import review_risk_targeted_unlearning_pilot

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shadow_summary = root / "shadow-summary.json"
            forget_index_file = root / "forget-members-k2.txt"
            matched_index_file = root / "matched-nonmembers-k2.txt"
            forget_index_file.write_text("10\n11\n", encoding="utf-8")
            matched_index_file.write_text("20\n21\n", encoding="utf-8")
            _write_json(
                shadow_summary,
                {
                    "status": "ready",
                    "mode": "loss-score-export",
                    "artifact_paths": {
                        "shadow_specs": [{"name": "shadow-01"}],
                    },
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
            )

            received_allowlists: list[list[int] | None] = []
            received_checkpoint_dirs: list[str] = []
            received_noise_seeds: list[int | None] = []

            def fake_extract(**kwargs):
                output_path = Path(kwargs["output_path"])
                records_path = Path(kwargs["records_path"])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                records_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"tensor")
                records_path.write_text("{}", encoding="utf-8")
                received_allowlists.append(kwargs.get("sample_id_allowlist"))
                received_checkpoint_dirs.append(str(kwargs.get("checkpoint_dir")))
                received_noise_seeds.append(kwargs.get("noise_seed"))
                return {
                    "status": "ready",
                    "output_path": str(output_path),
                    "records_path": str(records_path),
                    "checkpoint_dir": str(kwargs.get("checkpoint_dir")),
                    "sample_count": 2,
                    "score_stats": {"mean": 0.1, "min": 0.0, "max": 0.2},
                }

            def fake_eval(*, workspace, packet_summary, evaluation_style="threshold-transfer", provenance_status="workspace-verified"):
                del workspace, evaluation_style, provenance_status
                summary_name = Path(packet_summary).name
                auc = 0.62 if "baseline" in summary_name else 0.58
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

            with patch("diffaudit.defenses.risk_targeted_unlearning._extract_gsa_loss_scores", side_effect=fake_extract), patch(
                "diffaudit.defenses.risk_targeted_unlearning.evaluate_gsa_loss_score_packet",
                side_effect=fake_eval,
            ):
                summary = review_risk_targeted_unlearning_pilot(
                    workspace=root / "review",
                    shadow_reference_summary=shadow_summary,
                    target_member_dataset_dir=root / "target-member",
                    target_nonmember_dataset_dir=root / "target-nonmember",
                    baseline_checkpoint_root=root / "baseline-root",
                    baseline_checkpoint_dir=root / "baseline-root" / "checkpoint-1",
                    defended_checkpoint_dir=root / "defended-checkpoint",
                    forget_member_index_file=forget_index_file,
                    matched_nonmember_index_file=matched_index_file,
                    noise_seed=7,
                )

        self.assertEqual(summary["status"], "ready")
        self.assertEqual(summary["subset"]["sample_id_allowlist"], [10, 11, 20, 21])
        self.assertEqual(summary["subset"]["noise_seed"], 7)
        self.assertEqual(received_allowlists[0], [10, 11, 20, 21])
        self.assertEqual(received_noise_seeds, [7, 7, 7, 7])
        self.assertEqual(summary["comparison"]["target_transfer_delta"]["auc"], -0.04)
        self.assertIn(str(root / "defended-checkpoint"), received_checkpoint_dirs)

    def test_review_without_subset_files_uses_full_split(self) -> None:
        from diffaudit.defenses.risk_targeted_unlearning import review_risk_targeted_unlearning_pilot

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shadow_summary = root / "shadow-summary.json"
            _write_json(
                shadow_summary,
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
            )

            received_allowlists: list[list[int] | None] = []
            received_noise_seeds: list[int | None] = []

            def fake_extract(**kwargs):
                output_path = Path(kwargs["output_path"])
                records_path = Path(kwargs["records_path"])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                records_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"tensor")
                records_path.write_text("{}", encoding="utf-8")
                received_allowlists.append(kwargs.get("sample_id_allowlist"))
                received_noise_seeds.append(kwargs.get("noise_seed"))
                return {
                    "status": "ready",
                    "output_path": str(output_path),
                    "records_path": str(records_path),
                    "checkpoint_dir": str(kwargs.get("checkpoint_dir")),
                    "sample_count": 4,
                    "score_stats": {"mean": 0.1, "min": 0.0, "max": 0.2},
                }

            def fake_eval(*, workspace, packet_summary, evaluation_style="threshold-transfer", provenance_status="workspace-verified"):
                del workspace, packet_summary, evaluation_style, provenance_status
                return {
                    "status": "ready",
                    "target_transfer": {"metrics": {"auc": 0.6, "asr": 0.5, "tpr_at_1pct_fpr": 0.1, "tpr_at_0_1pct_fpr": 0.02}},
                    "target_self_diagnostic": {"metrics": {"auc": 0.6, "asr": 0.5, "tpr_at_1pct_fpr": 0.1, "tpr_at_0_1pct_fpr": 0.02}},
                    "shadow_pool": {"metrics": {"auc": 0.5}},
                }

            with patch("diffaudit.defenses.risk_targeted_unlearning._extract_gsa_loss_scores", side_effect=fake_extract), patch(
                "diffaudit.defenses.risk_targeted_unlearning.evaluate_gsa_loss_score_packet",
                side_effect=fake_eval,
            ):
                summary = review_risk_targeted_unlearning_pilot(
                    workspace=root / "review",
                    shadow_reference_summary=shadow_summary,
                    target_member_dataset_dir=root / "target-member",
                    target_nonmember_dataset_dir=root / "target-nonmember",
                    baseline_checkpoint_root=root / "baseline-root",
                    baseline_checkpoint_dir=root / "baseline-root" / "checkpoint-1",
                    defended_checkpoint_dir=root / "defended-checkpoint",
                    noise_seed=11,
                )

        self.assertIsNone(summary["subset"]["sample_id_allowlist"])
        self.assertEqual(summary["subset"]["noise_seed"], 11)
        self.assertEqual(received_allowlists[0], None)
        self.assertEqual(received_noise_seeds, [11, 11, 11, 11])

    def test_cli_runs_review_wrapper(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            forget_index_file = root / "forget-members-k1.txt"
            matched_index_file = root / "matched-nonmembers-k1.txt"
            shadow_summary = root / "shadow-summary.json"
            forget_index_file.write_text("10\n", encoding="utf-8")
            matched_index_file.write_text("20\n", encoding="utf-8")
            _write_json(
                shadow_summary,
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
            )

            def fake_review(**kwargs):
                return {
                    "status": "ready",
                    "artifact_paths": {"summary": str(root / "review" / "summary.json")},
                    "subset": {"sample_id_allowlist": [10, 20], "noise_seed": 5},
                    "comparison": {"target_transfer_delta": {"auc": -0.02}},
                }

            stdout = StringIO()
            with patch(
                "diffaudit.defenses.risk_targeted_unlearning.review_risk_targeted_unlearning_pilot",
                side_effect=fake_review,
            ):
                with redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "review-risk-targeted-unlearning-pilot",
                            "--workspace",
                            str(root / "review"),
                            "--shadow-reference-summary",
                            str(shadow_summary),
                            "--defended-checkpoint-dir",
                            str(root / "defended-checkpoint"),
                            "--forget-member-index-file",
                            str(forget_index_file),
                            "--matched-nonmember-index-file",
                            str(matched_index_file),
                            "--noise-seed",
                            "5",
                        ]
                    )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["subset"]["sample_id_allowlist"], [10, 20])
        self.assertEqual(payload["subset"]["noise_seed"], 5)


if __name__ == "__main__":
    unittest.main()
