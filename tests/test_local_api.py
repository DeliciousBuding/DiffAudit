import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


class LocalApiTests(unittest.TestCase):
    def test_cli_can_start_local_api_server(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            experiments_root = root / "experiments"
            jobs_root = root / "jobs"

            with patch("uvicorn.run") as mocked_run:
                stdout = StringIO()
                with redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "serve-local-api",
                            "--host",
                            "127.0.0.1",
                            "--port",
                            "8877",
                            "--experiments-root",
                            str(experiments_root),
                            "--jobs-root",
                            str(jobs_root),
                        ]
                    )

        self.assertEqual(exit_code, 0)
        mocked_run.assert_called_once()
        args, kwargs = mocked_run.call_args
        self.assertEqual(args[0], "diffaudit.local_api.app:create_app")
        self.assertEqual(kwargs["host"], "127.0.0.1")
        self.assertEqual(kwargs["port"], 8877)
        self.assertEqual(
            kwargs["factory"],
            True,
        )
        self.assertEqual(kwargs["reload"], False)
        self.assertEqual(kwargs["env_file"], None)

    def test_app_exposes_minimal_endpoints(self) -> None:
        from fastapi.testclient import TestClient

        from diffaudit.local_api.app import create_app

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            experiments_root = root / "experiments"
            experiments_root.mkdir(parents=True, exist_ok=True)
            jobs_root = root / "jobs"
            jobs_root.mkdir(parents=True, exist_ok=True)

            app = create_app(
                experiments_root=experiments_root,
                jobs_root=jobs_root,
            )
            client = TestClient(app)

            health_response = client.get("/health")
            self.assertEqual(health_response.status_code, 200)
            self.assertEqual(health_response.json()["status"], "ok")

            models_response = client.get("/api/v1/models")
            self.assertEqual(models_response.status_code, 200)
            model_keys = {item["key"] for item in models_response.json()}
            self.assertIn("sd15-ddim", model_keys)
            self.assertIn("kandinsky-v22", model_keys)
            self.assertIn("dit-xl2-256", model_keys)

    def test_recon_best_and_workspace_summary_read_experiment_summaries(self) -> None:
        from fastapi.testclient import TestClient

        from diffaudit.local_api.app import create_app

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            experiments_root = root / "experiments"
            jobs_root = root / "jobs"
            experiments_root.mkdir(parents=True, exist_ok=True)
            jobs_root.mkdir(parents=True, exist_ok=True)

            best_workspace = experiments_root / "recon-runtime-mainline-ddim-public-25-step10"
            best_workspace.mkdir(parents=True, exist_ok=True)
            summary_payload = {
                "status": "ready",
                "track": "black-box",
                "method": "recon",
                "paper": "BlackBox_Reconstruction_ArXiv2023",
                "mode": "runtime-mainline",
                "workspace": str(best_workspace),
                "runtime": {
                    "backend": "stable_diffusion",
                    "scheduler": "ddim",
                },
                "metrics": {
                    "auc": 0.768,
                    "asr": 0.52,
                    "tpr_at_1pct_fpr": 0.96,
                },
                "artifact_paths": {
                    "summary": str(best_workspace / "summary.json"),
                },
            }
            (best_workspace / "summary.json").write_text(
                json.dumps(summary_payload),
                encoding="utf-8",
            )

            blackbox_status_workspace = experiments_root / "blackbox-status"
            blackbox_status_workspace.mkdir(parents=True, exist_ok=True)
            (blackbox_status_workspace / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "track": "black-box",
                        "methods": {
                            "recon": {
                                "paper": "BlackBox_Reconstruction_ArXiv2023",
                                "best_evidence_mode": "runtime-mainline",
                                "best_evidence_path": str(best_workspace / "summary.json"),
                                "headline_metrics": {
                                    "auc": 0.768,
                                    "asr": 0.52,
                                    "tpr_at_1pct_fpr": 0.96,
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            app = create_app(
                experiments_root=experiments_root,
                jobs_root=jobs_root,
            )
            client = TestClient(app)

            best_response = client.get("/api/v1/experiments/recon/best")
            self.assertEqual(best_response.status_code, 200)
            best_payload = best_response.json()
            self.assertEqual(best_payload["workspace"], str(best_workspace))
            self.assertEqual(best_payload["metrics"]["auc"], 0.768)
            self.assertEqual(best_payload["backend"], "stable_diffusion")
            self.assertEqual(best_payload["scheduler"], "ddim")

            workspace_response = client.get(
                "/api/v1/experiments/recon-runtime-mainline-ddim-public-25-step10/summary"
            )
            self.assertEqual(workspace_response.status_code, 200)
            workspace_payload = workspace_response.json()
            self.assertEqual(workspace_payload["workspace"], str(best_workspace))
            self.assertEqual(workspace_payload["metrics"]["tpr_at_1pct_fpr"], 0.96)

    def test_create_and_get_job_persists_local_job_record(self) -> None:
        from fastapi.testclient import TestClient

        from diffaudit.local_api.app import create_app

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            experiments_root = root / "experiments"
            jobs_root = root / "jobs"
            experiments_root.mkdir(parents=True, exist_ok=True)
            jobs_root.mkdir(parents=True, exist_ok=True)

            app = create_app(
                experiments_root=experiments_root,
                jobs_root=jobs_root,
            )
            client = TestClient(app)

            create_response = client.post(
                "/api/v1/audit/jobs",
                json={
                    "job_type": "recon_artifact_mainline",
                    "workspace_name": "api-job-001",
                    "artifact_dir": "D:/artifacts/recon-scores",
                    "repo_root": "D:/Code/DiffAudit/Project/external/Reconstruction-based-Attack",
                },
            )
            self.assertEqual(create_response.status_code, 202)
            created = create_response.json()
            self.assertEqual(created["status"], "queued")
            self.assertEqual(created["job_type"], "recon_artifact_mainline")
            self.assertTrue(created["job_id"])

            job_file = jobs_root / f"{created['job_id']}.json"
            self.assertTrue(job_file.exists())

            get_response = client.get(f"/api/v1/audit/jobs/{created['job_id']}")
            self.assertEqual(get_response.status_code, 200)
            fetched = get_response.json()
            self.assertEqual(fetched["job_id"], created["job_id"])
            self.assertEqual(fetched["workspace_name"], "api-job-001")
            self.assertEqual(fetched["payload"]["artifact_dir"], "D:/artifacts/recon-scores")


if __name__ == "__main__":
    unittest.main()
