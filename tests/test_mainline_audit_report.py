import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


class MainlineAuditReportTests(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_builds_mainline_audit_report_with_gpu_recommendation(self) -> None:
        from diffaudit.reports.mainline_audit import build_mainline_audit_report

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table_path = root / "workspaces" / "implementation" / "artifacts" / "unified-attack-defense-table.json"
            self._write_json(
                table_path,
                {
                    "schema": "diffaudit.attack_defense_table.v1",
                    "updated_at": "2026-04-14T12:00:00+08:00",
                    "rows": [
                        {
                            "track": "black-box",
                            "attack": "recon DDIM public-100 step30",
                            "defense": "none",
                            "model": "Stable Diffusion v1.5 + DDIM",
                            "auc": 0.849,
                            "asr": 0.51,
                            "evidence_level": "runtime-mainline",
                            "note": "current black-box main evidence",
                            "boundary": "controlled / public-subset / risk-exists",
                            "source": "experiments/recon-runtime-mainline-ddim-public-100-step30/summary.json",
                        },
                        {
                            "track": "gray-box",
                            "attack": "PIA GPU512 baseline",
                            "defense": "none",
                            "model": "CIFAR-10 DDPM",
                            "auc": 0.841339,
                            "asr": 0.786133,
                            "evidence_level": "runtime-mainline",
                            "note": "baseline",
                            "boundary": "workspace-verified + adaptive-reviewed",
                            "source": "workspaces/gray-box/runs/pia-baseline/summary.json",
                        },
                        {
                            "track": "gray-box",
                            "attack": "PIA GPU512 baseline",
                            "defense": "provisional G-1 = stochastic-dropout (all_steps)",
                            "model": "CIFAR-10 DDPM",
                            "auc": 0.828075,
                            "asr": 0.767578,
                            "evidence_level": "runtime-mainline",
                            "note": "defended",
                            "boundary": "workspace-verified + adaptive-reviewed",
                            "source": "workspaces/gray-box/runs/pia-defense/summary.json",
                        },
                        {
                            "track": "gray-box",
                            "attack": "SecMI GPU full split",
                            "defense": "none",
                            "model": "CIFAR-10 DDPM",
                            "auc": 0.885833,
                            "asr": 0.8154,
                            "evidence_level": "gpu-full-mainline",
                            "note": "gray-box corroboration",
                            "boundary": "workspace-verified + full local split execution",
                            "source": "workspaces/gray-box/runs/secmi-full/summary.json",
                        },
                        {
                            "track": "white-box",
                            "attack": "GSA 1k-3shadow",
                            "defense": "none",
                            "model": "CIFAR-10 DDPM",
                            "auc": 0.998192,
                            "asr": 0.9895,
                            "evidence_level": "runtime-mainline",
                            "note": "attack",
                            "boundary": "admitted bridge / not final benchmark",
                            "source": "workspaces/white-box/runs/gsa/summary.json",
                        },
                        {
                            "track": "white-box",
                            "attack": "GSA 1k-3shadow",
                            "defense": "W-1 strong-v3 full-scale",
                            "model": "DPDM / Diffusion-DP",
                            "auc": 0.488783,
                            "asr": 0.4985,
                            "evidence_level": "runtime-smoke",
                            "note": "defended",
                            "boundary": "admitted bridge / not final benchmark",
                            "source": "workspaces/white-box/runs/w1/summary.json",
                        },
                    ],
                },
            )

            for rel in (
                "experiments/recon-runtime-mainline-ddim-public-100-step30/summary.json",
                "workspaces/gray-box/runs/pia-baseline/summary.json",
                "workspaces/gray-box/runs/pia-defense/summary.json",
                "workspaces/gray-box/runs/secmi-full/summary.json",
                "workspaces/white-box/runs/gsa/summary.json",
                "workspaces/white-box/runs/w1/summary.json",
            ):
                self._write_json(root / rel, {"status": "ready"})

            self._write_json(
                root / "workspaces" / "gray-box" / "runs" / "pia-cifar10-runtime-mainline-20260410-gpu-128-adaptive" / "summary.json",
                {
                    "track": "gray-box",
                    "method": "pia",
                    "mode": "runtime-mainline",
                    "workspace_name": "pia-cifar10-runtime-mainline-20260410-gpu-128-adaptive",
                    "sample_count_per_split": 128,
                    "artifact_paths": {"summary": "workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260410-gpu-128-adaptive/summary.json"},
                    "defense": {"enabled": False, "dropout_activation_schedule": "off"},
                    "metrics": {"auc": 0.82, "asr": 0.76},
                    "adaptive_check": {"metrics": {"auc": 0.82, "asr": 0.76}},
                    "cost": {"wall_clock_seconds": 50.0},
                },
            )
            self._write_json(
                root / "workspaces" / "gray-box" / "runs" / "pia-cifar10-runtime-mainline-dropout-defense-20260410-gpu-128-allsteps-adaptive" / "summary.json",
                {
                    "track": "gray-box",
                    "method": "pia",
                    "mode": "runtime-mainline",
                    "workspace_name": "pia-cifar10-runtime-mainline-dropout-defense-20260410-gpu-128-allsteps-adaptive",
                    "sample_count_per_split": 128,
                    "artifact_paths": {"summary": "workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260410-gpu-128-allsteps-adaptive/summary.json"},
                    "defense": {"enabled": True, "dropout_activation_schedule": "all_steps"},
                    "metrics": {"auc": 0.81, "asr": 0.75},
                    "adaptive_check": {"metrics": {"auc": 0.81, "asr": 0.75}},
                    "cost": {"wall_clock_seconds": 55.0},
                },
            )
            self._write_json(
                root / "workspaces" / "gray-box" / "runs" / "pia-cifar10-runtime-mainline-dropout-defense-20260410-gpu-128-latesteps-adaptive" / "summary.json",
                {
                    "track": "gray-box",
                    "method": "pia",
                    "mode": "runtime-mainline",
                    "workspace_name": "pia-cifar10-runtime-mainline-dropout-defense-20260410-gpu-128-latesteps-adaptive",
                    "sample_count_per_split": 128,
                    "artifact_paths": {"summary": "workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260410-gpu-128-latesteps-adaptive/summary.json"},
                    "defense": {"enabled": True, "dropout_activation_schedule": "late_steps_only"},
                    "metrics": {"auc": 0.818, "asr": 0.762},
                    "adaptive_check": {"metrics": {"auc": 0.818, "asr": 0.762}},
                    "cost": {"wall_clock_seconds": 56.0},
                },
            )
            self._write_json(
                root / "workspaces" / "gray-box" / "runs" / "pia-cifar10-runtime-mainline-20260410-cpu-32-adaptive" / "summary.json",
                {
                    "track": "gray-box",
                    "method": "pia",
                    "mode": "runtime-mainline",
                    "workspace_name": "pia-cifar10-runtime-mainline-20260410-cpu-32-adaptive",
                    "sample_count_per_split": 32,
                    "artifact_paths": {"summary": "workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260410-cpu-32-adaptive/summary.json"},
                    "defense": {"enabled": False, "dropout_activation_schedule": "off"},
                    "metrics": {"auc": 0.80, "asr": 0.75},
                    "adaptive_check": {"metrics": {"auc": 0.80, "asr": 0.75}},
                    "cost": {"wall_clock_seconds": 20.0, "device": "cpu"},
                    "device": "cpu",
                },
            )
            self._write_json(
                root / "workspaces" / "gray-box" / "runs" / "pia-cifar10-runtime-mainline-dropout-defense-20260410-cpu-32-allsteps-adaptive" / "summary.json",
                {
                    "track": "gray-box",
                    "method": "pia",
                    "mode": "runtime-mainline",
                    "workspace_name": "pia-cifar10-runtime-mainline-dropout-defense-20260410-cpu-32-allsteps-adaptive",
                    "sample_count_per_split": 32,
                    "artifact_paths": {"summary": "workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260410-cpu-32-allsteps-adaptive/summary.json"},
                    "defense": {"enabled": True, "dropout_activation_schedule": "all_steps"},
                    "metrics": {"auc": 0.76, "asr": 0.70},
                    "adaptive_check": {"metrics": {"auc": 0.76, "asr": 0.70}},
                    "cost": {"wall_clock_seconds": 22.0, "device": "cpu"},
                    "device": "cpu",
                },
            )

            report = build_mainline_audit_report(
                research_root=root,
                workspace=root / "workspaces" / "implementation" / "reports" / "mainline-audit",
            )
            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["system_posture"]["overall_risk_level"], "critical")
            self.assertEqual(report["gpu_next_action"]["status"], "recommended")
            self.assertIn("GPU128", report["gpu_next_action"]["recommended_rung"])
            self.assertIn("gpu-128", report["gpu_next_action"]["baseline_summary"])
            self.assertEqual(report["dropout_schedule_recommendation"]["recommended_schedule"], "all_steps")
            self.assertEqual(report["gray_box_runtime_health"]["sample_count_per_split"], 128)
            self.assertIn("gray-box", report["corroborating_evidence"])
            self.assertEqual(report["corroborating_evidence"]["gray-box"][0]["attack"], "SecMI GPU full split")
            self.assertEqual(
                report["gray_box_runtime_health"]["schedule_stability"]["all_steps"]["run_count"],
                1,
            )
            self.assertTrue(report["deployment_blockers"])
            self.assertIn("operator", report["consumer_views"])
            self.assertTrue((root / "workspaces" / "implementation" / "reports" / "mainline-audit" / "summary.json").exists())
            self.assertTrue((root / "workspaces" / "implementation" / "reports" / "mainline-audit" / "next_gpu_action.json").exists())

    def test_cli_summarizes_mainline_audit_report(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table_path = root / "workspaces" / "implementation" / "artifacts" / "unified-attack-defense-table.json"
            self._write_json(
                table_path,
                {
                    "schema": "diffaudit.attack_defense_table.v1",
                    "updated_at": "2026-04-14T12:00:00+08:00",
                    "rows": [
                        {
                            "track": "black-box",
                            "attack": "recon DDIM public-100 step30",
                            "defense": "none",
                            "model": "Stable Diffusion v1.5 + DDIM",
                            "auc": 0.849,
                            "asr": 0.51,
                            "evidence_level": "runtime-mainline",
                            "note": "current black-box main evidence",
                            "boundary": "controlled / public-subset / risk-exists",
                            "source": "experiments/recon-runtime-mainline-ddim-public-100-step30/summary.json",
                        },
                        {
                            "track": "gray-box",
                            "attack": "PIA GPU512 baseline",
                            "defense": "none",
                            "model": "CIFAR-10 DDPM",
                            "auc": 0.84,
                            "asr": 0.78,
                            "evidence_level": "runtime-mainline",
                            "note": "baseline",
                            "boundary": "workspace-verified + adaptive-reviewed",
                            "source": "workspaces/gray-box/runs/pia-baseline/summary.json",
                        },
                        {
                            "track": "gray-box",
                            "attack": "PIA GPU512 baseline",
                            "defense": "provisional G-1 = stochastic-dropout (all_steps)",
                            "model": "CIFAR-10 DDPM",
                            "auc": 0.83,
                            "asr": 0.76,
                            "evidence_level": "runtime-mainline",
                            "note": "defended",
                            "boundary": "workspace-verified + adaptive-reviewed",
                            "source": "workspaces/gray-box/runs/pia-defense/summary.json",
                        },
                        {
                            "track": "gray-box",
                            "attack": "SecMI GPU full split",
                            "defense": "none",
                            "model": "CIFAR-10 DDPM",
                            "auc": 0.885833,
                            "asr": 0.8154,
                            "evidence_level": "gpu-full-mainline",
                            "note": "gray-box corroboration",
                            "boundary": "workspace-verified + full local split execution",
                            "source": "workspaces/gray-box/runs/secmi-full/summary.json",
                        },
                        {
                            "track": "white-box",
                            "attack": "GSA 1k-3shadow",
                            "defense": "none",
                            "model": "CIFAR-10 DDPM",
                            "auc": 0.998,
                            "asr": 0.989,
                            "evidence_level": "runtime-mainline",
                            "note": "attack",
                            "boundary": "admitted bridge / not final benchmark",
                            "source": "workspaces/white-box/runs/gsa/summary.json",
                        },
                        {
                            "track": "white-box",
                            "attack": "GSA 1k-3shadow",
                            "defense": "W-1 strong-v3 full-scale",
                            "model": "DPDM / Diffusion-DP",
                            "auc": 0.49,
                            "asr": 0.49,
                            "evidence_level": "runtime-smoke",
                            "note": "defended",
                            "boundary": "admitted bridge / not final benchmark",
                            "source": "workspaces/white-box/runs/w1/summary.json",
                        },
                    ],
                },
            )

            for rel in (
                "experiments/recon-runtime-mainline-ddim-public-100-step30/summary.json",
                "workspaces/gray-box/runs/pia-baseline/summary.json",
                "workspaces/gray-box/runs/pia-defense/summary.json",
                "workspaces/gray-box/runs/secmi-full/summary.json",
                "workspaces/white-box/runs/gsa/summary.json",
                "workspaces/white-box/runs/w1/summary.json",
            ):
                self._write_json(root / rel, {"status": "ready"})

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "summarize-mainline-audit",
                        "--research-root",
                        str(root),
                        "--workspace",
                        str(root / "reports" / "mainline-audit"),
                        "--attack-defense-table",
                        str(table_path),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["mode"], "mainline-audit-report")
        self.assertEqual(payload["system_posture"]["overall_risk_level"], "critical")
        self.assertEqual(payload["schema"], "diffaudit.mainline_audit_report.v2")
        self.assertEqual(payload["corroborating_evidence"]["gray-box"][0]["attack"], "SecMI GPU full split")


if __name__ == "__main__":
    unittest.main()
