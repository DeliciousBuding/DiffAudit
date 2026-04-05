import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


class BlackBoxReportTests(unittest.TestCase):
    def test_builds_blackbox_status_report(self) -> None:
        from diffaudit.reports.blackbox_status import build_blackbox_status_report

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            experiments_root = root / "experiments"
            (experiments_root / "clid-artifact-summary").mkdir(parents=True)
            (experiments_root / "recon-upstream-eval-smoke").mkdir(parents=True)
            (experiments_root / "recon-mainline-smoke").mkdir(parents=True)
            (experiments_root / "variation-synth-smoke").mkdir(parents=True)

            (experiments_root / "clid-artifact-summary" / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "track": "black-box",
                        "method": "clid",
                        "paper": "CLiD_NeurIPS2024",
                        "mode": "artifact-summary",
                        "metrics": {
                            "target": {
                                "auc": 0.96,
                                "best_asr": 0.89,
                                "tpr_at_1pct_fpr": 0.49,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            (experiments_root / "recon-upstream-eval-smoke" / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "track": "black-box",
                        "method": "recon",
                        "paper": "BlackBox_Reconstruction_ArXiv2023",
                        "mode": "upstream-eval-smoke",
                        "metrics": {
                            "reported_accuracy": 1.0,
                            "reported_auc": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (experiments_root / "recon-mainline-smoke" / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "track": "black-box",
                        "method": "recon",
                        "paper": "BlackBox_Reconstruction_ArXiv2023",
                        "mode": "mainline-smoke",
                        "metrics": {
                            "auc": 1.0,
                            "asr": 1.0,
                            "tpr_at_1pct_fpr": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (experiments_root / "variation-synth-smoke" / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "track": "black-box",
                        "method": "variation",
                        "paper": "Towards_Black_Box_Diffusion_2024",
                        "mode": "synthetic-smoke",
                        "metrics": {
                            "auc": 1.0,
                            "asr": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = build_blackbox_status_report(
                experiments_root=experiments_root,
                workspace=root / "blackbox-status",
            )

            self.assertTrue((root / "blackbox-status" / "summary.json").exists())

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["track"], "black-box")
        self.assertEqual(report["method_count"], 3)
        self.assertIn("clid", report["methods"])
        self.assertEqual(report["methods"]["clid"]["headline_metrics"]["auc"], 0.96)
        self.assertEqual(report["methods"]["recon"]["best_evidence_mode"], "mainline-smoke")

    def test_cli_summarizes_blackbox_results(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            experiments_root = root / "experiments"
            (experiments_root / "variation-synth-smoke").mkdir(parents=True)
            (experiments_root / "variation-synth-smoke" / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "track": "black-box",
                        "method": "variation",
                        "paper": "Towards_Black_Box_Diffusion_2024",
                        "mode": "synthetic-smoke",
                        "metrics": {
                            "auc": 1.0,
                            "asr": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "summarize-blackbox-results",
                        "--experiments-root",
                        str(experiments_root),
                        "--workspace",
                        str(root / "blackbox-status"),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["method_count"], 1)
        self.assertEqual(payload["methods"]["variation"]["headline_metrics"]["auc"], 1.0)

    def test_ignores_aggregate_summary_without_method(self) -> None:
        from diffaudit.reports.blackbox_status import build_blackbox_status_report

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            experiments_root = root / "experiments"
            (experiments_root / "variation-synth-smoke").mkdir(parents=True)
            (experiments_root / "blackbox-status").mkdir(parents=True)

            (experiments_root / "variation-synth-smoke" / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "track": "black-box",
                        "method": "variation",
                        "paper": "Towards_Black_Box_Diffusion_2024",
                        "mode": "synthetic-smoke",
                        "metrics": {
                            "auc": 1.0,
                            "asr": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (experiments_root / "blackbox-status" / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "track": "black-box",
                        "method_count": 1,
                        "methods": {},
                    }
                ),
                encoding="utf-8",
            )

            report = build_blackbox_status_report(
                experiments_root=experiments_root,
                workspace=root / "out",
            )

        self.assertEqual(report["method_count"], 1)
        self.assertIn("variation", report["methods"])

    def test_artifact_mainline_outranks_mainline_smoke(self) -> None:
        from diffaudit.reports.blackbox_status import build_blackbox_status_report

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            experiments_root = root / "experiments"
            (experiments_root / "recon-mainline-smoke").mkdir(parents=True)
            (experiments_root / "recon-artifact-mainline").mkdir(parents=True)

            (experiments_root / "recon-mainline-smoke" / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "track": "black-box",
                        "method": "recon",
                        "paper": "BlackBox_Reconstruction_ArXiv2023",
                        "mode": "mainline-smoke",
                        "metrics": {
                            "auc": 1.0,
                            "asr": 1.0,
                            "tpr_at_1pct_fpr": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (experiments_root / "recon-artifact-mainline" / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "track": "black-box",
                        "method": "recon",
                        "paper": "BlackBox_Reconstruction_ArXiv2023",
                        "mode": "artifact-mainline",
                        "metrics": {
                            "auc": 1.0,
                            "asr": 1.0,
                            "tpr_at_1pct_fpr": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = build_blackbox_status_report(
                experiments_root=experiments_root,
                workspace=root / "out",
            )

        self.assertEqual(report["methods"]["recon"]["best_evidence_mode"], "artifact-mainline")

    def test_runtime_mainline_outranks_artifact_mainline(self) -> None:
        from diffaudit.reports.blackbox_status import build_blackbox_status_report

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            experiments_root = root / "experiments"
            (experiments_root / "recon-artifact-mainline").mkdir(parents=True)
            (experiments_root / "recon-runtime-mainline-kandinsky").mkdir(parents=True)

            (experiments_root / "recon-artifact-mainline" / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "track": "black-box",
                        "method": "recon",
                        "paper": "BlackBox_Reconstruction_ArXiv2023",
                        "mode": "artifact-mainline",
                        "metrics": {
                            "auc": 0.91,
                            "asr": 0.75,
                            "tpr_at_1pct_fpr": 0.31,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (experiments_root / "recon-runtime-mainline-kandinsky" / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "track": "black-box",
                        "method": "recon",
                        "paper": "BlackBox_Reconstruction_ArXiv2023",
                        "mode": "runtime-mainline",
                        "metrics": {
                            "auc": 0.92,
                            "asr": 0.76,
                            "tpr_at_1pct_fpr": 0.32,
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = build_blackbox_status_report(
                experiments_root=experiments_root,
                workspace=root / "out",
            )

        self.assertEqual(report["methods"]["recon"]["best_evidence_mode"], "runtime-mainline")

    def test_larger_runtime_mainline_outranks_smaller_runtime_mainline(self) -> None:
        from diffaudit.reports.blackbox_status import build_blackbox_status_report

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            experiments_root = root / "experiments"
            (experiments_root / "recon-runtime-mainline-kandinsky").mkdir(parents=True)
            (experiments_root / "recon-runtime-mainline-ddim-public-10").mkdir(parents=True)

            (experiments_root / "recon-runtime-mainline-kandinsky" / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "track": "black-box",
                        "method": "recon",
                        "paper": "BlackBox_Reconstruction_ArXiv2023",
                        "mode": "runtime-mainline",
                        "metrics": {"auc": 1.0, "asr": 0.5, "tpr_at_1pct_fpr": 0.0},
                        "artifacts": {
                            "target_member": {"sample_count": 1},
                            "target_non_member": {"sample_count": 1},
                            "shadow_member": {"sample_count": 1},
                            "shadow_non_member": {"sample_count": 1},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (experiments_root / "recon-runtime-mainline-ddim-public-10" / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "track": "black-box",
                        "method": "recon",
                        "paper": "BlackBox_Reconstruction_ArXiv2023",
                        "mode": "runtime-mainline",
                        "metrics": {"auc": 0.82, "asr": 0.55, "tpr_at_1pct_fpr": 1.0},
                        "artifacts": {
                            "target_member": {"sample_count": 10},
                            "target_non_member": {"sample_count": 10},
                            "shadow_member": {"sample_count": 10},
                            "shadow_non_member": {"sample_count": 10},
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = build_blackbox_status_report(
                experiments_root=experiments_root,
                workspace=root / "out",
            )

        self.assertEqual(
            report["methods"]["recon"]["best_evidence_path"],
            str(experiments_root / "recon-runtime-mainline-ddim-public-10" / "summary.json"),
        )


if __name__ == "__main__":
    unittest.main()
