import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


RECON_CONFIG_TEMPLATE = """
task:
  name: recon-plan
  model_family: diffusion
  access_level: black_box
assets:
  dataset_id: toy-dataset
  dataset_root: PLACEHOLDER_DATASET
  model_id: target-lora
  model_dir: PLACEHOLDER_MODEL
attack:
  method: recon
  num_samples: 8
  parameters:
    pretrained_model_name_or_path: runwayml/stable-diffusion-v1-5
    num_validation_images: 3
    inference_steps: 30
    eval_method: threshold
report:
  output_dir: experiments/recon-plan
"""


def create_minimal_recon_repo(repo_root: Path) -> None:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "inference.py").write_text(
        "from diffusers import StableDiffusionPipeline\n"
        "parser = None\n",
        encoding="utf-8",
    )
    (repo_root / "cal_embedding.py").write_text(
        "from transformers import DeiTFeatureExtractor\n"
        "def compute_scores():\n    return None\n",
        encoding="utf-8",
    )
    (repo_root / "test_accuracy.py").write_text(
        "import argparse\n"
        "import torch\n"
        "\n"
        "class DefineClassifier:\n    pass\n"
        "\n"
        "def parse_args():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--target_member_dir')\n"
        "    parser.add_argument('--target_non_member_dir')\n"
        "    parser.add_argument('--shadow_member_dir')\n"
        "    parser.add_argument('--shadow_non_member_dir')\n"
        "    parser.add_argument('--method', default='threshold')\n"
        "    return parser.parse_args()\n"
        "\n"
        "def process_data():\n"
        "    return None\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    args = parse_args()\n"
        "    target_member = torch.load(args.target_member_dir)\n"
        "    target_non_member = torch.load(args.target_non_member_dir)\n"
        "    print(f'Accuracy with Best Threshold: {0.75:.2f}')\n"
        "    print(f'AUC-ROC: {1.0:.3f}')\n",
        encoding="utf-8",
    )
    (repo_root / "train_text_to_image_lora.py").write_text("# train", encoding="utf-8")


class ReconAttackTests(unittest.TestCase):
    def test_cli_plans_recon(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "query.pt"
            dataset_path.write_text("dataset", encoding="utf-8")
            model_dir = root / "lora"
            model_dir.mkdir()
            config_path = root / "audit.yaml"
            config_path.write_text(
                RECON_CONFIG_TEMPLATE
                .replace("PLACEHOLDER_DATASET", str(dataset_path).replace("\\", "/"))
                .replace("PLACEHOLDER_MODEL", str(model_dir).replace("\\", "/")),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["plan-recon", "--config", str(config_path)])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["inference_entrypoint"], "inference.py")
        self.assertEqual(payload["evaluation_entrypoint"], "test_accuracy.py")
        self.assertEqual(payload["inference_steps"], 30)

    def test_cli_probes_recon_assets(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "query.pt"
            dataset_path.write_text("dataset", encoding="utf-8")
            model_dir = root / "lora"
            model_dir.mkdir()
            config_path = root / "audit.yaml"
            config_path.write_text(
                RECON_CONFIG_TEMPLATE
                .replace("PLACEHOLDER_DATASET", str(dataset_path).replace("\\", "/"))
                .replace("PLACEHOLDER_MODEL", str(model_dir).replace("\\", "/")),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["probe-recon-assets", "--config", str(config_path)])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["checks"]["dataset_root"])
        self.assertTrue(payload["checks"]["model_dir"])

    def test_cli_runs_recon_dry_run(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "query.pt"
            dataset_path.write_text("dataset", encoding="utf-8")
            model_dir = root / "lora"
            model_dir.mkdir()
            repo_root = root / "recon"
            create_minimal_recon_repo(repo_root)

            config_path = root / "audit.yaml"
            config_path.write_text(
                RECON_CONFIG_TEMPLATE
                .replace("PLACEHOLDER_DATASET", str(dataset_path).replace("\\", "/"))
                .replace("PLACEHOLDER_MODEL", str(model_dir).replace("\\", "/")),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "dry-run-recon",
                        "--config",
                        str(config_path),
                        "--repo-root",
                        str(repo_root),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["checks"]["inference_has_pipeline"])
        self.assertTrue(payload["checks"]["evaluation_has_classifier"])

    def test_cli_runs_recon_eval_smoke(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "run-recon-eval-smoke",
                        "--workspace",
                        str(root / "recon-eval-smoke"),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["mode"], "eval-smoke")
        self.assertIn("auc", payload["metrics"])
        self.assertFalse((root / "recon-eval-smoke" / "synthetic-score-artifacts").exists())

    def test_summarize_recon_artifacts_writes_summary(self) -> None:
        from diffaudit.attacks.recon import summarize_recon_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_dir = root / "scores"
            artifact_dir.mkdir()
            torch_payloads = {
                "target_member.pt": [([0.91], 1), ([0.88], 1), ([0.86], 1)],
                "target_non_member.pt": [([0.18], 0), ([0.22], 0), ([0.25], 0)],
                "shadow_member.pt": [([0.89], 1), ([0.84], 1), ([0.83], 1)],
                "shadow_non_member.pt": [([0.17], 0), ([0.21], 0), ([0.24], 0)],
            }
            for filename, payload in torch_payloads.items():
                import torch

                torch.save(payload, artifact_dir / filename)

            result = summarize_recon_artifacts(
                artifact_dir=artifact_dir,
                workspace=root / "recon-artifact-summary",
            )

            self.assertTrue((root / "recon-artifact-summary" / "summary.json").exists())

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["mode"], "artifact-summary")
        self.assertIn("shadow_auc", result["metrics"])
        self.assertIn("target_auc", result["metrics"])

    def test_cli_summarizes_recon_artifacts(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_dir = root / "scores"
            artifact_dir.mkdir()
            torch_payloads = {
                "target_member.pt": [([0.91], 1), ([0.88], 1), ([0.86], 1)],
                "target_non_member.pt": [([0.18], 0), ([0.22], 0), ([0.25], 0)],
                "shadow_member.pt": [([0.89], 1), ([0.84], 1), ([0.83], 1)],
                "shadow_non_member.pt": [([0.17], 0), ([0.21], 0), ([0.24], 0)],
            }
            for filename, payload in torch_payloads.items():
                import torch

                torch.save(payload, artifact_dir / filename)

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "summarize-recon-artifacts",
                        "--artifact-dir",
                        str(artifact_dir),
                        "--workspace",
                        str(root / "recon-artifact-summary"),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["mode"], "artifact-summary")
        self.assertIn("target_auc", payload["metrics"])

    def test_run_recon_upstream_eval_smoke_writes_summary(self) -> None:
        from diffaudit.attacks.recon import run_recon_upstream_eval_smoke

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "recon"
            create_minimal_recon_repo(repo_root)
            result = run_recon_upstream_eval_smoke(
                workspace=root / "recon-upstream-eval-smoke",
                repo_root=repo_root,
                method="threshold",
            )
            self.assertTrue((root / "recon-upstream-eval-smoke" / "summary.json").exists())
            self.assertFalse((root / "recon-upstream-eval-smoke" / "synthetic-score-artifacts").exists())

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["mode"], "upstream-eval-smoke")
        self.assertEqual(result["evaluation_method"], "threshold")
        self.assertTrue(result["checks"]["upstream_script_succeeded"])

    def test_cli_runs_recon_upstream_eval_smoke(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "recon"
            create_minimal_recon_repo(repo_root)

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "run-recon-upstream-eval-smoke",
                        "--workspace",
                        str(root / "recon-upstream-eval-smoke"),
                        "--repo-root",
                        str(repo_root),
                        "--method",
                        "threshold",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["mode"], "upstream-eval-smoke")
        self.assertTrue(payload["checks"]["upstream_script_succeeded"])

    def test_run_recon_mainline_smoke_writes_unified_summary(self) -> None:
        from diffaudit.attacks.recon import run_recon_mainline_smoke

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "recon"
            create_minimal_recon_repo(repo_root)

            result = run_recon_mainline_smoke(
                workspace=root / "recon-mainline-smoke",
                repo_root=repo_root,
                method="threshold",
            )

            workspace = root / "recon-mainline-smoke"
            self.assertTrue((workspace / "summary.json").exists())
            self.assertTrue((workspace / "eval-smoke" / "summary.json").exists())
            self.assertTrue((workspace / "artifact-summary" / "summary.json").exists())
            self.assertTrue((workspace / "upstream-eval-smoke" / "summary.json").exists())

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["mode"], "mainline-smoke")
        self.assertTrue(result["checks"]["all_stages_ready"])
        self.assertEqual(result["stages"]["artifact_summary"]["status"], "ready")
        self.assertEqual(result["stages"]["upstream_eval_smoke"]["status"], "ready")
        self.assertEqual(result["metrics"]["auc"], 1.0)

    def test_cli_runs_recon_mainline_smoke(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "recon"
            create_minimal_recon_repo(repo_root)

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "run-recon-mainline-smoke",
                        "--workspace",
                        str(root / "recon-mainline-smoke"),
                        "--repo-root",
                        str(repo_root),
                        "--method",
                        "threshold",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["mode"], "mainline-smoke")
        self.assertTrue(payload["checks"]["all_stages_ready"])
        self.assertEqual(payload["stages"]["eval_smoke"]["status"], "ready")

    def test_run_recon_artifact_mainline_writes_unified_summary(self) -> None:
        from diffaudit.attacks.recon import run_recon_artifact_mainline
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "recon"
            artifact_dir = root / "scores"
            create_minimal_recon_repo(repo_root)
            artifact_dir.mkdir()
            torch_payloads = {
                "target_member.pt": [([0.91], 1), ([0.88], 1), ([0.86], 1)],
                "target_non_member.pt": [([0.18], 0), ([0.22], 0), ([0.25], 0)],
                "shadow_member.pt": [([0.89], 1), ([0.84], 1), ([0.83], 1)],
                "shadow_non_member.pt": [([0.17], 0), ([0.21], 0), ([0.24], 0)],
            }
            for filename, payload in torch_payloads.items():
                torch.save(payload, artifact_dir / filename)

            result = run_recon_artifact_mainline(
                artifact_dir=artifact_dir,
                workspace=root / "recon-artifact-mainline",
                repo_root=repo_root,
                method="threshold",
            )

            workspace = root / "recon-artifact-mainline"
            self.assertTrue((workspace / "summary.json").exists())
            self.assertTrue((workspace / "artifact-summary" / "summary.json").exists())
            self.assertTrue((workspace / "upstream-eval" / "summary.json").exists())

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["mode"], "artifact-mainline")
        self.assertTrue(result["checks"]["all_stages_ready"])
        self.assertEqual(result["stages"]["artifact_summary"]["status"], "ready")
        self.assertEqual(result["stages"]["upstream_eval"]["status"], "ready")
        self.assertEqual(result["metrics"]["auc"], 1.0)

    def test_cli_runs_recon_artifact_mainline(self) -> None:
        from diffaudit.cli import main
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "recon"
            artifact_dir = root / "scores"
            create_minimal_recon_repo(repo_root)
            artifact_dir.mkdir()
            torch_payloads = {
                "target_member.pt": [([0.91], 1), ([0.88], 1), ([0.86], 1)],
                "target_non_member.pt": [([0.18], 0), ([0.22], 0), ([0.25], 0)],
                "shadow_member.pt": [([0.89], 1), ([0.84], 1), ([0.83], 1)],
                "shadow_non_member.pt": [([0.17], 0), ([0.21], 0), ([0.24], 0)],
            }
            for filename, payload in torch_payloads.items():
                torch.save(payload, artifact_dir / filename)

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "run-recon-artifact-mainline",
                        "--artifact-dir",
                        str(artifact_dir),
                        "--workspace",
                        str(root / "recon-artifact-mainline"),
                        "--repo-root",
                        str(repo_root),
                        "--method",
                        "threshold",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["mode"], "artifact-mainline")
        self.assertTrue(payload["checks"]["all_stages_ready"])
        self.assertEqual(payload["stages"]["upstream_eval"]["status"], "ready")


if __name__ == "__main__":
    unittest.main()
