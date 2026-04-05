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
        "class DefineClassifier:\n    pass\n"
        "def process_data():\n    return None\n",
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


if __name__ == "__main__":
    unittest.main()
