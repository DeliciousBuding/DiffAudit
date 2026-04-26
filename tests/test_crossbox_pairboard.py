import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import torch


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


class CrossboxPairboardTests(unittest.TestCase):
    def test_summarize_repeated_holdout_runs_aggregates_candidate_stats(self) -> None:
        from diffaudit.attacks.crossbox_pairboard import _summarize_repeated_holdout_runs

        summary = _summarize_repeated_holdout_runs(
            [
                {
                    "run_index": 0,
                    "seed": 3,
                    "test": {
                        "best_single": {"selected_surface": "pia"},
                        "candidates": {
                            "best_single": {
                                "auc": 0.8,
                                "asr": 0.7,
                                "tpr_at_1pct_fpr": 0.2,
                                "tpr_at_0_1pct_fpr": 0.1,
                            },
                            "weighted_average": {
                                "auc": 0.82,
                                "asr": 0.72,
                                "tpr_at_1pct_fpr": 0.24,
                                "tpr_at_0_1pct_fpr": 0.12,
                            },
                            "logistic_2feature": {
                                "auc": 0.84,
                                "asr": 0.74,
                                "tpr_at_1pct_fpr": 0.28,
                                "tpr_at_0_1pct_fpr": 0.14,
                            },
                            "support_disconfirm_neutral": {
                                "auc": 0.79,
                                "asr": 0.69,
                                "tpr_at_1pct_fpr": 0.18,
                                "tpr_at_0_1pct_fpr": 0.08,
                            },
                        },
                    },
                },
                {
                    "run_index": 1,
                    "seed": 5,
                    "test": {
                        "best_single": {"selected_surface": "gsa"},
                        "candidates": {
                            "best_single": {
                                "auc": 0.83,
                                "asr": 0.73,
                                "tpr_at_1pct_fpr": 0.21,
                                "tpr_at_0_1pct_fpr": 0.09,
                            },
                            "weighted_average": {
                                "auc": 0.81,
                                "asr": 0.71,
                                "tpr_at_1pct_fpr": 0.2,
                                "tpr_at_0_1pct_fpr": 0.11,
                            },
                            "logistic_2feature": {
                                "auc": 0.86,
                                "asr": 0.75,
                                "tpr_at_1pct_fpr": 0.29,
                                "tpr_at_0_1pct_fpr": 0.16,
                            },
                            "support_disconfirm_neutral": {
                                "auc": 0.8,
                                "asr": 0.7,
                                "tpr_at_1pct_fpr": 0.19,
                                "tpr_at_0_1pct_fpr": 0.07,
                            },
                        },
                    },
                },
            ]
        )

        self.assertEqual(summary["repeats"], 2)
        self.assertEqual(summary["selected_surface_counts"], {"pia": 1, "gsa": 1})
        self.assertEqual(summary["candidate_metrics"]["weighted_average"]["auc"]["mean"], 0.815)
        self.assertEqual(summary["candidate_metrics"]["weighted_average"]["auc"]["std"], 0.005)
        self.assertEqual(summary["best_single_comparison"]["weighted_average"]["auc"]["win_count"], 1)
        self.assertEqual(summary["best_single_comparison"]["weighted_average"]["auc"]["loss_count"], 1)
        self.assertEqual(summary["best_single_comparison"]["logistic_2feature"]["tpr_at_1pct_fpr"]["win_count"], 2)

    def test_load_pairboard_surface_recovers_gsa_sample_ids_from_records(self) -> None:
        from diffaudit.attacks.crossbox_pairboard import load_pairboard_surface

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scores_root = root / "loss-scores"
            scores_root.mkdir(parents=True, exist_ok=True)

            target_member = scores_root / "target_member-loss-scores.pt"
            target_nonmember = scores_root / "target_non_member-loss-scores.pt"
            target_member_records = scores_root / "target_member-loss-scores.jsonl"
            target_nonmember_records = scores_root / "target_non_member-loss-scores.jsonl"
            torch.save(torch.tensor([0.81, 0.74], dtype=torch.float32), target_member)
            torch.save(torch.tensor([0.22, 0.31], dtype=torch.float32), target_nonmember)
            target_member_records.write_text(
                "\n".join(
                    [
                        json.dumps({"sample_index": 0, "sample_path": "00-data_batch_1-00965.png", "loss_score": 0.81}),
                        json.dumps({"sample_index": 1, "sample_path": "00-data_batch_1-01278.png", "loss_score": 0.74}),
                    ]
                ),
                encoding="utf-8",
            )
            target_nonmember_records.write_text(
                "\n".join(
                    [
                        json.dumps({"sample_index": 0, "sample_path": "00-data_batch_1-00467.png", "loss_score": 0.22}),
                        json.dumps({"sample_index": 1, "sample_path": "00-data_batch_1-02287.png", "loss_score": 0.31}),
                    ]
                ),
                encoding="utf-8",
            )

            summary_path = root / "summary.json"
            _write_json(
                summary_path,
                {
                    "status": "ready",
                    "mode": "loss-score-export",
                    "exports": {
                        "target_member": {
                            "output_path": str(target_member),
                            "records_path": str(target_member_records),
                        },
                        "target_non_member": {
                            "output_path": str(target_nonmember),
                            "records_path": str(target_nonmember_records),
                        },
                    },
                },
            )

            surface = load_pairboard_surface(summary_path, name="gsa-loss-packet")

        self.assertEqual(surface["member_indices"], [965, 1278])
        self.assertEqual(surface["nonmember_indices"], [467, 2287])

    def test_load_pairboard_surface_supports_gsa_loss_score_export_summary(self) -> None:
        from diffaudit.attacks.crossbox_pairboard import load_pairboard_surface

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scores_root = root / "loss-scores"
            scores_root.mkdir(parents=True, exist_ok=True)

            target_member = scores_root / "target_member-loss-scores.pt"
            target_nonmember = scores_root / "target_non_member-loss-scores.pt"
            torch.save(torch.tensor([0.81, 0.74, 0.69], dtype=torch.float32), target_member)
            torch.save(torch.tensor([0.22, 0.31, 0.36], dtype=torch.float32), target_nonmember)

            summary_path = root / "summary.json"
            _write_json(
                summary_path,
                {
                    "status": "ready",
                    "mode": "loss-score-export",
                    "exports": {
                        "target_member": {"output_path": str(target_member)},
                        "target_non_member": {"output_path": str(target_nonmember)},
                    },
                },
            )

            surface = load_pairboard_surface(summary_path, name="gsa-loss-packet")

        self.assertEqual(surface["name"], "gsa-loss-packet")
        self.assertEqual(surface["source_kind"], "gsa-loss-score-export")
        self.assertEqual(surface["member_scores"].tolist(), [0.81, 0.74, 0.69])
        self.assertEqual(surface["nonmember_scores"].tolist(), [0.22, 0.31, 0.36])
        self.assertIsNone(surface["member_indices"])
        self.assertIsNone(surface["nonmember_indices"])

    def test_run_crossbox_pairboard_aligns_shared_indices_and_reports_candidates(self) -> None:
        from diffaudit.attacks.crossbox_pairboard import run_crossbox_pairboard

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            surface_a = root / "surface-a.json"
            surface_b = root / "surface-b.json"

            _write_json(
                surface_a,
                {
                    "member_scores": [0.93, 0.88, 0.85, 0.8, 0.78, 0.74],
                    "nonmember_scores": [0.41, 0.37, 0.34, 0.29, 0.25, 0.2],
                    "member_indices": [10, 11, 12, 13, 14, 15],
                    "nonmember_indices": [20, 21, 22, 23, 24, 25],
                },
            )
            _write_json(
                surface_b,
                {
                    "member_scores": [0.79, 0.91, 0.7, 0.76, 0.85, 0.73],
                    "nonmember_scores": [0.32, 0.4, 0.18, 0.27, 0.35, 0.23],
                    "member_indices": [12, 10, 15, 13, 11, 14],
                    "nonmember_indices": [22, 20, 25, 23, 21, 24],
                },
            )

            summary = run_crossbox_pairboard(
                workspace=root / "pairboard-run",
                surface_a_path=surface_a,
                surface_b_path=surface_b,
                surface_a_name="pia",
                surface_b_name="gsa",
                calibration_fraction=0.5,
                seed=7,
            )

            summary_written = Path(summary["artifact_paths"]["summary"]).exists()

        self.assertEqual(summary["status"], "ready")
        self.assertEqual(summary["mode"], "crossbox-pairboard")
        self.assertEqual(summary["pairboard"]["alignment"]["member_strategy"], "shared-index-intersection")
        self.assertEqual(summary["pairboard"]["member_indices"], [10, 11, 12, 13, 14, 15])
        self.assertEqual(summary["pairboard"]["nonmember_indices"], [20, 21, 22, 23, 24, 25])
        self.assertEqual(summary["pairboard"]["split"]["calibration_count_per_label"], 3)
        self.assertEqual(summary["pairboard"]["split"]["test_count_per_label"], 3)
        self.assertIn("best_single", summary["test"]["candidates"])
        self.assertIn("weighted_average", summary["test"]["candidates"])
        self.assertIn("logistic_2feature", summary["test"]["candidates"])
        self.assertIn("support_disconfirm_neutral", summary["test"]["candidates"])
        self.assertIn(
            summary["calibration"]["best_single"]["selected_surface"],
            {"pia", "gsa"},
        )
        self.assertTrue(summary_written)

    def test_run_crossbox_pairboard_handles_uneven_shared_overlap(self) -> None:
        from diffaudit.attacks.crossbox_pairboard import run_crossbox_pairboard

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            surface_a = root / "surface-a.json"
            surface_b = root / "surface-b.json"

            _write_json(
                surface_a,
                {
                    "member_scores": [0.91, 0.84, 0.8, 0.75, 0.71],
                    "nonmember_scores": [0.42, 0.39, 0.35, 0.31, 0.28, 0.24],
                    "member_indices": [10, 11, 12, 13, 14],
                    "nonmember_indices": [20, 21, 22, 23, 24, 25],
                },
            )
            _write_json(
                surface_b,
                {
                    "member_scores": [0.82, 0.89, 0.76, 0.7],
                    "nonmember_scores": [0.29, 0.4, 0.34, 0.22, 0.26],
                    "member_indices": [12, 10, 14, 99],
                    "nonmember_indices": [25, 20, 23, 200, 24],
                },
            )

            summary = run_crossbox_pairboard(
                workspace=root / "pairboard-uneven-run",
                surface_a_path=surface_a,
                surface_b_path=surface_b,
                surface_a_name="pia",
                surface_b_name="gsa",
                calibration_fraction=0.5,
                seed=11,
            )

        self.assertEqual(summary["pairboard"]["member_indices"], [10, 12, 14])
        self.assertEqual(summary["pairboard"]["nonmember_indices"], [20, 23, 24, 25])
        self.assertEqual(summary["pairboard"]["split"]["calibration_member_count"], 2)
        self.assertEqual(summary["pairboard"]["split"]["calibration_nonmember_count"], 2)
        self.assertEqual(summary["pairboard"]["split"]["test_member_count"], 1)
        self.assertEqual(summary["pairboard"]["split"]["test_nonmember_count"], 2)
        self.assertIn("weighted_average", summary["test"]["candidates"])

    def test_run_crossbox_pairboard_reports_repeated_holdout_summary(self) -> None:
        from diffaudit.attacks.crossbox_pairboard import run_crossbox_pairboard

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            surface_a = root / "surface-a.json"
            surface_b = root / "surface-b.json"

            _write_json(
                surface_a,
                {
                    "member_scores": [0.95, 0.91, 0.88, 0.83, 0.79, 0.75, 0.72, 0.69],
                    "nonmember_scores": [0.43, 0.4, 0.36, 0.33, 0.28, 0.24, 0.21, 0.17],
                    "member_indices": [10, 11, 12, 13, 14, 15, 16, 17],
                    "nonmember_indices": [20, 21, 22, 23, 24, 25, 26, 27],
                },
            )
            _write_json(
                surface_b,
                {
                    "member_scores": [0.84, 0.93, 0.8, 0.76, 0.87, 0.71, 0.68, 0.74],
                    "nonmember_scores": [0.31, 0.38, 0.19, 0.27, 0.35, 0.22, 0.15, 0.24],
                    "member_indices": [12, 10, 17, 15, 11, 14, 16, 13],
                    "nonmember_indices": [22, 20, 27, 23, 21, 24, 26, 25],
                },
            )

            summary = run_crossbox_pairboard(
                workspace=root / "pairboard-repeated-run",
                surface_a_path=surface_a,
                surface_b_path=surface_b,
                surface_a_name="pia",
                surface_b_name="gsa",
                calibration_fraction=0.5,
                seed=3,
                repeats=3,
            )

        repeated = summary["repeated_holdout"]
        self.assertEqual(repeated["configuration"]["repeats"], 3)
        self.assertEqual(repeated["configuration"]["seed_stride"], 2)
        self.assertEqual(len(repeated["runs"]), 3)
        self.assertEqual([run["seed"] for run in repeated["runs"]], [3, 5, 7])
        self.assertIn("weighted_average", repeated["aggregate"]["candidate_metrics"])
        self.assertEqual(
            set(repeated["aggregate"]["candidate_metrics"]["logistic_2feature"]["auc"].keys()),
            {"mean", "std", "min", "max"},
        )
        weighted_tail = repeated["aggregate"]["best_single_comparison"]["weighted_average"]["tpr_at_1pct_fpr"]
        self.assertEqual(weighted_tail["win_count"] + weighted_tail["tie_count"] + weighted_tail["loss_count"], 3)

    def test_run_crossbox_pairboard_reports_tail_gated_cascade(self) -> None:
        from diffaudit.attacks.crossbox_pairboard import run_crossbox_pairboard

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            surface_a = root / "surface-a.json"
            surface_b = root / "surface-b.json"

            _write_json(
                surface_a,
                {
                    "member_scores": [0.94, 0.9, 0.87, 0.84, 0.8, 0.77, 0.74, 0.71],
                    "nonmember_scores": [0.45, 0.4, 0.37, 0.33, 0.28, 0.24, 0.2, 0.17],
                    "member_indices": [10, 11, 12, 13, 14, 15, 16, 17],
                    "nonmember_indices": [20, 21, 22, 23, 24, 25, 26, 27],
                },
            )
            _write_json(
                surface_b,
                {
                    "member_scores": [0.81, 0.93, 0.78, 0.74, 0.86, 0.69, 0.66, 0.72],
                    "nonmember_scores": [0.29, 0.38, 0.21, 0.27, 0.34, 0.24, 0.16, 0.22],
                    "member_indices": [12, 10, 17, 15, 11, 14, 16, 13],
                    "nonmember_indices": [22, 20, 27, 23, 21, 24, 26, 25],
                },
            )

            summary = run_crossbox_pairboard(
                workspace=root / "pairboard-tail-gated-run",
                surface_a_path=surface_a,
                surface_b_path=surface_b,
                surface_a_name="pia",
                surface_b_name="gsa",
                calibration_fraction=0.5,
                seed=5,
                repeats=3,
                enable_tail_gated_cascade=True,
                cascade_anchor_name="gsa",
                cascade_candidate_name="logistic_2feature",
                cascade_route_fractions=[0.25, 0.5],
                cascade_gammas=[0.0, 0.2],
                cascade_secondary_cost_ratio=0.25,
            )

        self.assertIn("tail_gated_cascade", summary["test"]["candidates"])
        self.assertIn("tail_gated_cascade", summary["repeated_holdout"]["aggregate"]["candidate_metrics"])
        cascade_analysis = summary["analysis"]["tail_gated_cascade"]
        self.assertEqual(cascade_analysis["selection"]["anchor_name"], "gsa")
        self.assertEqual(cascade_analysis["selection"]["candidate_name"], "logistic_2feature")
        self.assertIn(cascade_analysis["selection"]["target_route_fraction"], [0.25, 0.5])
        self.assertIn(cascade_analysis["selection"]["gamma"], [0.0, 0.2])
        self.assertGreater(cascade_analysis["test"]["routed_count"], 0)
        self.assertIn(
            "tail_gated_cascade",
            summary["repeated_holdout"]["aggregate"]["best_single_comparison"],
        )

    def test_cli_analyzes_crossbox_pairboard_with_family_surface(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pia_scores = root / "pia-scores.json"
            tmiadm_family_scores = root / "tmiadm-family-scores.json"

            _write_json(
                pia_scores,
                {
                    "member_scores": [0.91, 0.87, 0.83, 0.79, 0.74, 0.7],
                    "nonmember_scores": [0.48, 0.42, 0.38, 0.31, 0.27, 0.22],
                },
            )
            _write_json(
                tmiadm_family_scores,
                {
                    "fused": {
                        "member_scores": [0.82, 0.8, 0.77, 0.73, 0.7, 0.66],
                        "nonmember_scores": [0.41, 0.36, 0.33, 0.27, 0.23, 0.19],
                    }
                },
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "analyze-crossbox-pairboard",
                        "--workspace",
                        str(root / "pairboard-cli"),
                        "--surface-a",
                        str(pia_scores),
                        "--surface-a-name",
                        "pia",
                        "--surface-b",
                        str(tmiadm_family_scores),
                        "--surface-b-name",
                        "tmiadm",
                        "--surface-b-family",
                        "fused",
                        "--calibration-fraction",
                        "0.5",
                        "--seed",
                        "5",
                    ]
                )

            payload = json.loads(stdout.getvalue())
            summary_written = Path(payload["artifact_paths"]["summary"]).exists()

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["mode"], "crossbox-pairboard")
        self.assertEqual(payload["surfaces"]["surface_b"]["family"], "fused")
        self.assertIn("logistic_2feature", payload["test"]["candidates"])
        self.assertTrue(summary_written)

    def test_cli_analyzes_crossbox_pairboard_with_repeated_holdout(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            surface_a = root / "surface-a.json"
            surface_b = root / "surface-b.json"

            _write_json(
                surface_a,
                {
                    "member_scores": [0.91, 0.88, 0.85, 0.82, 0.79, 0.76],
                    "nonmember_scores": [0.42, 0.39, 0.35, 0.29, 0.24, 0.2],
                    "member_indices": [10, 11, 12, 13, 14, 15],
                    "nonmember_indices": [20, 21, 22, 23, 24, 25],
                },
            )
            _write_json(
                surface_b,
                {
                    "member_scores": [0.8, 0.9, 0.72, 0.77, 0.85, 0.74],
                    "nonmember_scores": [0.3, 0.4, 0.19, 0.28, 0.36, 0.23],
                    "member_indices": [12, 10, 15, 13, 11, 14],
                    "nonmember_indices": [22, 20, 25, 23, 21, 24],
                },
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "analyze-crossbox-pairboard",
                        "--workspace",
                        str(root / "pairboard-cli-repeated"),
                        "--surface-a",
                        str(surface_a),
                        "--surface-a-name",
                        "pia",
                        "--surface-b",
                        str(surface_b),
                        "--surface-b-name",
                        "gsa",
                        "--calibration-fraction",
                        "0.5",
                        "--seed",
                        "9",
                        "--repeats",
                        "3",
                    ]
                )

            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["repeated_holdout"]["configuration"]["repeats"], 3)
        self.assertEqual(len(payload["repeated_holdout"]["runs"]), 3)

    def test_cli_analyzes_crossbox_pairboard_with_tail_gated_cascade(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            surface_a = root / "surface-a.json"
            surface_b = root / "surface-b.json"

            _write_json(
                surface_a,
                {
                    "member_scores": [0.91, 0.88, 0.85, 0.82, 0.79, 0.76],
                    "nonmember_scores": [0.42, 0.39, 0.35, 0.29, 0.24, 0.2],
                    "member_indices": [10, 11, 12, 13, 14, 15],
                    "nonmember_indices": [20, 21, 22, 23, 24, 25],
                },
            )
            _write_json(
                surface_b,
                {
                    "member_scores": [0.8, 0.9, 0.72, 0.77, 0.85, 0.74],
                    "nonmember_scores": [0.3, 0.4, 0.19, 0.28, 0.36, 0.23],
                    "member_indices": [12, 10, 15, 13, 11, 14],
                    "nonmember_indices": [22, 20, 25, 23, 21, 24],
                },
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "analyze-crossbox-pairboard",
                        "--workspace",
                        str(root / "pairboard-cli-tail-gated"),
                        "--surface-a",
                        str(surface_a),
                        "--surface-a-name",
                        "pia",
                        "--surface-b",
                        str(surface_b),
                        "--surface-b-name",
                        "gsa",
                        "--calibration-fraction",
                        "0.5",
                        "--seed",
                        "9",
                        "--tail-gated-cascade",
                        "--cascade-anchor-name",
                        "gsa",
                        "--cascade-candidate-name",
                        "logistic_2feature",
                        "--cascade-route-fractions",
                        "0.25,0.5",
                        "--cascade-gammas",
                        "0,0.2",
                        "--cascade-secondary-cost-ratio",
                        "0.25",
                    ]
                )

            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertIn("tail_gated_cascade", payload["test"]["candidates"])
        self.assertEqual(payload["analysis"]["tail_gated_cascade"]["selection"]["anchor_name"], "gsa")


if __name__ == "__main__":
    unittest.main()
