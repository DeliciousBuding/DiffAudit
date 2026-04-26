import json
import tempfile
import unittest
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


class GrayboxTriscoreCanaryTests(unittest.TestCase):
    def test_load_identity_aligned_surface_reuses_pia_indices_when_runtime_contract_matches(self) -> None:
        from diffaudit.attacks.graybox_triscore import load_identity_aligned_surface

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pia_scores = root / "pia-scores.json"
            pia_summary = root / "pia-summary.json"
            tmiadm_family_scores = root / "tmiadm-family-scores.json"
            tmiadm_summary = root / "tmiadm-summary.json"

            _write_json(
                pia_scores,
                {
                    "member_scores": [0.9, 0.8, 0.7, 0.6],
                    "nonmember_scores": [0.4, 0.3, 0.2, 0.1],
                    "member_indices": [10, 11, 12, 13],
                    "nonmember_indices": [20, 21, 22, 23],
                },
            )
            _write_json(
                pia_summary,
                {
                    "runtime": {
                        "dataset_root": "dataset-A",
                        "member_split_root": "split-A",
                        "model_dir": "model-A",
                        "max_samples": 4,
                        "num_samples": 4,
                    },
                    "sample_count_per_split": 4,
                },
            )
            _write_json(
                tmiadm_family_scores,
                {
                    "long_window": {
                        "member_scores": [1.1, 0.8, 0.75, 0.55],
                        "nonmember_scores": [0.6, 0.35, 0.3, 0.2],
                        "metrics": {
                            "auc": 0.75,
                            "asr": 0.75,
                            "threshold": 0.5,
                            "tpr_at_1pct_fpr": 0.25,
                            "tpr_at_0_1pct_fpr": 0.25,
                        },
                    }
                },
            )
            _write_json(
                tmiadm_summary,
                {
                    "runtime": {
                        "dataset_root": "dataset-A",
                        "member_split_root": "split-A",
                        "model_dir": "model-A",
                        "max_samples": 4,
                        "num_samples": 4,
                    },
                    "sample_count_per_split": 4,
                },
            )

            surface = load_identity_aligned_surface(
                name="gpu256_undefended",
                pia_scores_path=pia_scores,
                pia_summary_path=pia_summary,
                tmiadm_family_scores_path=tmiadm_family_scores,
                tmiadm_summary_path=tmiadm_summary,
                tmiadm_family="long_window",
            )

        self.assertEqual(surface["member_indices"], [10, 11, 12, 13])
        self.assertEqual(surface["nonmember_indices"], [20, 21, 22, 23])
        self.assertEqual(surface["analysis"]["identity_alignment"]["status"], "frozen")
        self.assertEqual(
            surface["analysis"]["identity_alignment"]["tmiadm_index_strategy"],
            "adopt-pia-order",
        )

    def test_load_identity_aligned_surface_rejects_runtime_contract_mismatch(self) -> None:
        from diffaudit.attacks.graybox_triscore import load_identity_aligned_surface

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pia_scores = root / "pia-scores.json"
            pia_summary = root / "pia-summary.json"
            tmiadm_family_scores = root / "tmiadm-family-scores.json"
            tmiadm_summary = root / "tmiadm-summary.json"

            _write_json(
                pia_scores,
                {
                    "member_scores": [0.9, 0.8],
                    "nonmember_scores": [0.2, 0.1],
                    "member_indices": [10, 11],
                    "nonmember_indices": [20, 21],
                },
            )
            _write_json(
                pia_summary,
                {
                    "runtime": {
                        "dataset_root": "dataset-A",
                        "member_split_root": "split-A",
                        "model_dir": "model-A",
                        "max_samples": 2,
                        "num_samples": 2,
                    },
                    "sample_count_per_split": 2,
                },
            )
            _write_json(
                tmiadm_family_scores,
                {
                    "long_window": {
                        "member_scores": [0.8, 0.7],
                        "nonmember_scores": [0.3, 0.2],
                        "metrics": {
                            "auc": 0.75,
                            "asr": 0.75,
                            "threshold": 0.5,
                            "tpr_at_1pct_fpr": 0.5,
                            "tpr_at_0_1pct_fpr": 0.5,
                        },
                    }
                },
            )
            _write_json(
                tmiadm_summary,
                {
                    "runtime": {
                        "dataset_root": "dataset-A",
                        "member_split_root": "split-B",
                        "model_dir": "model-A",
                        "max_samples": 2,
                        "num_samples": 2,
                    },
                    "sample_count_per_split": 2,
                },
            )

            with self.assertRaisesRegex(ValueError, "member_split_root"):
                load_identity_aligned_surface(
                    name="gpu256_undefended",
                    pia_scores_path=pia_scores,
                    pia_summary_path=pia_summary,
                    tmiadm_family_scores_path=tmiadm_family_scores,
                    tmiadm_summary_path=tmiadm_summary,
                    tmiadm_family="long_window",
                )

    def test_run_graybox_triscore_canary_emits_required_summary_schema(self) -> None:
        from diffaudit.attacks.graybox_triscore import run_graybox_triscore_canary

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            contract_reference = root / "contract-reference.json"
            _write_json(
                contract_reference,
                {
                    "contract": {
                        "name": "cdi-paired-canary-summary",
                        "version": "2026-04-17-r3-contract",
                        "feature_mode": "paired-pia-secmi-control-z-linear",
                        "component_reporting_required": True,
                        "headline_use_allowed": False,
                        "external_evidence_allowed": False,
                    }
                },
            )

            def write_surface(prefix: str, pia_member, pia_nonmember, tmi_member, tmi_nonmember) -> tuple[Path, Path, Path, Path]:
                pia_scores = root / f"{prefix}-pia-scores.json"
                pia_summary = root / f"{prefix}-pia-summary.json"
                tmiadm_family_scores = root / f"{prefix}-tmiadm-family-scores.json"
                tmiadm_summary = root / f"{prefix}-tmiadm-summary.json"
                _write_json(
                    pia_scores,
                    {
                        "member_scores": pia_member,
                        "nonmember_scores": pia_nonmember,
                        "member_indices": list(range(100, 100 + len(pia_member))),
                        "nonmember_indices": list(range(200, 200 + len(pia_nonmember))),
                    },
                )
                _write_json(
                    pia_summary,
                    {
                        "runtime": {
                            "dataset_root": "dataset-A",
                            "member_split_root": "split-A",
                            "model_dir": "model-A",
                            "max_samples": len(pia_member),
                            "num_samples": len(pia_member),
                        },
                        "sample_count_per_split": len(pia_member),
                    },
                )
                _write_json(
                    tmiadm_family_scores,
                    {
                        "long_window": {
                            "member_scores": tmi_member,
                            "nonmember_scores": tmi_nonmember,
                            "metrics": {
                                "auc": 0.75,
                                "asr": 0.75,
                                "threshold": 0.5,
                                "tpr_at_1pct_fpr": 0.25,
                                "tpr_at_0_1pct_fpr": 0.25,
                            },
                        }
                    },
                )
                _write_json(
                    tmiadm_summary,
                    {
                        "runtime": {
                            "dataset_root": "dataset-A",
                            "member_split_root": "split-A",
                            "model_dir": "model-A",
                            "max_samples": len(tmi_member),
                            "num_samples": len(tmi_member),
                        },
                        "sample_count_per_split": len(tmi_member),
                    },
                )
                return pia_scores, pia_summary, tmiadm_family_scores, tmiadm_summary

            surface_a = write_surface(
                "undefended",
                [0.99, 0.95, 0.87, 0.8, 0.78, 0.72],
                [0.62, 0.54, 0.48, 0.46, 0.4, 0.35],
                [0.91, 0.85, 0.83, 0.74, 0.72, 0.7],
                [0.66, 0.57, 0.51, 0.43, 0.38, 0.32],
            )
            surface_b = write_surface(
                "defended",
                [0.96, 0.92, 0.81, 0.77, 0.74, 0.69],
                [0.68, 0.61, 0.55, 0.52, 0.49, 0.42],
                [0.88, 0.84, 0.78, 0.71, 0.66, 0.62],
                [0.71, 0.64, 0.58, 0.54, 0.5, 0.46],
            )

            summary = run_graybox_triscore_canary(
                run_root=root / "run",
                contract_reference_path=contract_reference,
                surfaces=[
                    {
                        "name": "gpu256_undefended",
                        "pia_scores_path": surface_a[0],
                        "pia_summary_path": surface_a[1],
                        "tmiadm_family_scores_path": surface_a[2],
                        "tmiadm_summary_path": surface_a[3],
                        "tmiadm_family": "long_window",
                    },
                    {
                        "name": "gpu256_defended",
                        "pia_scores_path": surface_b[0],
                        "pia_summary_path": surface_b[1],
                        "tmiadm_family_scores_path": surface_b[2],
                        "tmiadm_summary_path": surface_b[3],
                        "tmiadm_family": "long_window",
                    },
                ],
            )

            written_summary = json.loads((root / "run" / "audit_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["contract"]["name"], "cdi-paired-canary-summary")
        self.assertTrue(summary["contract"]["component_reporting_required"])
        self.assertFalse(summary["contract"]["headline_use_allowed"])
        self.assertFalse(summary["contract"]["external_evidence_allowed"])
        self.assertIn(summary["verdict"], {"positive", "bounded", "no-go"})
        self.assertIn("auc", summary["metrics"])
        self.assertIn("asr", summary["metrics"])
        self.assertIn("tpr_at_1pct_fpr", summary["metrics"])
        self.assertIn("tpr_at_0_1pct_fpr", summary["metrics"])
        self.assertIn("component_auc", summary["metrics"])
        self.assertIn("pia", summary["metrics"]["component_auc"])
        self.assertIn("tmiadm", summary["metrics"]["component_auc"])
        self.assertIn("zscore_sum", summary["metrics"]["component_auc"])
        self.assertIn("identity_alignment", summary["analysis"])
        self.assertIn("feature_orientations", summary["analysis"])
        self.assertIn("tri_scorer_details", summary["analysis"])
        self.assertEqual(len(summary["surfaces"]), 2)
        self.assertEqual(written_summary["metrics"], summary["metrics"])


if __name__ == "__main__":
    unittest.main()
