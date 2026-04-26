import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np


def _write_json(path: Path, payload: dict) -> None:
    import json

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


class TemporalLrTests(unittest.TestCase):
    def test_evaluate_temporal_lr_packets_reports_primary_and_sensitivity_candidates(self) -> None:
        from diffaudit.attacks.temporal_lr import evaluate_temporal_lr_packets

        rng = np.random.default_rng(11)
        feature_names = [
            "eps_abs_mean_early",
            "eps_abs_mean_late",
            "eps_abs_late_over_early",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            calibration_packet = root / "calibration-feature-packet.json"
            transfer_packet = root / "transfer-feature-packet.json"

            calibration_member = np.column_stack(
                [
                    rng.normal(0.8, 0.05, size=64),
                    rng.normal(1.2, 0.08, size=64),
                    rng.normal(1.45, 0.15, size=64),
                ]
            )
            calibration_nonmember = np.column_stack(
                [
                    rng.normal(0.82, 0.05, size=64),
                    rng.normal(0.7, 0.08, size=64),
                    rng.normal(0.86, 0.15, size=64),
                ]
            )
            transfer_member = np.column_stack(
                [
                    rng.normal(0.81, 0.06, size=96),
                    rng.normal(1.15, 0.1, size=96),
                    rng.normal(1.4, 0.16, size=96),
                ]
            )
            transfer_nonmember = np.column_stack(
                [
                    rng.normal(0.83, 0.06, size=96),
                    rng.normal(0.73, 0.1, size=96),
                    rng.normal(0.89, 0.16, size=96),
                ]
            )

            _write_json(
                calibration_packet,
                {
                    "feature_names": feature_names,
                    "member_indices": list(range(1000, 1064)),
                    "nonmember_indices": list(range(2000, 2064)),
                    "member_features": np.round(calibration_member, 6).tolist(),
                    "nonmember_features": np.round(calibration_nonmember, 6).tolist(),
                },
            )
            _write_json(
                transfer_packet,
                {
                    "feature_names": feature_names,
                    "member_indices": list(range(3000, 3096)),
                    "nonmember_indices": list(range(4000, 4096)),
                    "member_features": np.round(transfer_member, 6).tolist(),
                    "nonmember_features": np.round(transfer_nonmember, 6).tolist(),
                },
            )

            result = evaluate_temporal_lr_packets(
                workspace=root / "temporal-lr-eval",
                calibration_feature_packet=calibration_packet,
                transfer_feature_packet=transfer_packet,
                cv_splits=4,
                cv_repeats=2,
                random_seed=3,
            )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["mode"], "temporal-lr-calibration-transfer")
        self.assertEqual(result["calibration"]["primary_candidate"], "eps_abs_mean_late")
        self.assertEqual(result["calibration"]["sensitivity_candidate"], "eps_abs_late_over_early")
        self.assertGreater(
            result["calibration"]["candidates"]["eps_abs_mean_late"]["metrics"]["auc"],
            0.9,
        )
        self.assertGreater(
            result["transfer"]["metrics"]["auc"],
            0.9,
        )
        self.assertIn("shift_audit", result["transfer"])

    def test_cli_evaluates_temporal_lr_packets(self) -> None:
        from diffaudit.cli import main

        rng = np.random.default_rng(13)
        feature_names = [
            "eps_abs_mean_early",
            "eps_abs_mean_late",
            "eps_abs_late_over_early",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            calibration_packet = root / "calibration-feature-packet.json"

            member = np.column_stack(
                [
                    rng.normal(0.8, 0.05, size=48),
                    rng.normal(1.1, 0.08, size=48),
                    rng.normal(1.35, 0.15, size=48),
                ]
            )
            nonmember = np.column_stack(
                [
                    rng.normal(0.82, 0.05, size=48),
                    rng.normal(0.72, 0.08, size=48),
                    rng.normal(0.9, 0.15, size=48),
                ]
            )
            _write_json(
                calibration_packet,
                {
                    "feature_names": feature_names,
                    "member_indices": list(range(48)),
                    "nonmember_indices": list(range(100, 148)),
                    "member_features": np.round(member, 6).tolist(),
                    "nonmember_features": np.round(nonmember, 6).tolist(),
                },
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "evaluate-temporal-lr-packets",
                        "--workspace",
                        str(root / "temporal-lr-cli"),
                        "--calibration-feature-packet",
                        str(calibration_packet),
                        "--cv-splits",
                        "4",
                        "--cv-repeats",
                        "2",
                        "--random-seed",
                        "5",
                    ]
                )
            import json

            payload = json.loads(stdout.getvalue())
            summary_written = Path(payload["artifact_paths"]["summary"]).exists()

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["mode"], "temporal-lr-calibration-transfer")
        self.assertIn("eps_abs_mean_late", payload["calibration"]["candidates"])
        self.assertTrue(summary_written)


if __name__ == "__main__":
    unittest.main()
