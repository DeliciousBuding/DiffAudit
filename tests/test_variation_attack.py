import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


VARIATION_CONFIG_TEMPLATE = """
task:
  name: variation-plan
  model_family: diffusion
  access_level: black_box
assets:
  dataset_id: variation-queries
  dataset_root: PLACEHOLDER_QUERY_ROOT
  model_id: hosted-image-variation-api
attack:
  method: variation
  num_samples: 8
  parameters:
    endpoint: https://example.invalid/variation
    num_averages: 3
    distance_metric: l2
    threshold_strategy: threshold
report:
  output_dir: experiments/variation-plan
"""


class VariationAttackTests(unittest.TestCase):
    def test_cli_plans_variation(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            query_root = root / "queries"
            query_root.mkdir()
            (query_root / "q1.png").write_bytes(b"png")

            config_path = root / "audit.yaml"
            config_path.write_text(
                VARIATION_CONFIG_TEMPLATE.replace(
                    "PLACEHOLDER_QUERY_ROOT",
                    str(query_root).replace("\\", "/"),
                ),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["plan-variation", "--config", str(config_path)])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["entrypoint"], "variation-api")
        self.assertEqual(payload["num_averages"], 3)
        self.assertEqual(payload["distance_metric"], "l2")

    def test_cli_probes_variation_assets(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            query_root = root / "queries"
            query_root.mkdir()
            (query_root / "q1.png").write_bytes(b"png")

            config_path = root / "audit.yaml"
            config_path.write_text(
                VARIATION_CONFIG_TEMPLATE.replace(
                    "PLACEHOLDER_QUERY_ROOT",
                    str(query_root).replace("\\", "/"),
                ),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["probe-variation-assets", "--config", str(config_path)])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["checks"]["query_image_root"])
        self.assertTrue(payload["checks"]["query_images_present"])
        self.assertFalse(payload["checks"]["paper_eval_layout"])
        self.assertEqual(payload["layout"]["flat_query_count"], 1)

    def test_cli_probes_variation_assets_with_member_nonmember_layout(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            query_root = root / "queries"
            member_root = query_root / "member"
            nonmember_root = query_root / "nonmember"
            member_root.mkdir(parents=True)
            nonmember_root.mkdir(parents=True)
            (member_root / "m1.png").write_bytes(b"png")
            (nonmember_root / "n1.png").write_bytes(b"png")

            config_path = root / "audit.yaml"
            config_path.write_text(
                VARIATION_CONFIG_TEMPLATE.replace(
                    "PLACEHOLDER_QUERY_ROOT",
                    str(query_root).replace("\\", "/"),
                ),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["probe-variation-assets", "--config", str(config_path)])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["checks"]["member_split_present"])
        self.assertTrue(payload["checks"]["nonmember_split_present"])
        self.assertTrue(payload["checks"]["paper_eval_layout"])
        self.assertEqual(payload["layout"]["member_query_count"], 1)
        self.assertEqual(payload["layout"]["nonmember_query_count"], 1)

    def test_cli_runs_variation_dry_run(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            query_root = root / "queries"
            query_root.mkdir()
            (query_root / "q1.png").write_bytes(b"png")

            config_path = root / "audit.yaml"
            config_path.write_text(
                VARIATION_CONFIG_TEMPLATE.replace(
                    "PLACEHOLDER_QUERY_ROOT",
                    str(query_root).replace("\\", "/"),
                ),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["dry-run-variation", "--config", str(config_path)])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["checks"]["distance_metric_supported"])
        self.assertTrue(payload["checks"]["threshold_strategy_supported"])

    def test_cli_runs_variation_synthetic_smoke(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "run-variation-synth-smoke",
                        "--workspace",
                        str(root / "variation-synth-smoke"),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["mode"], "synthetic-smoke")
        self.assertIn("auc", payload["metrics"])
        self.assertFalse((root / "variation-synth-smoke" / "synthetic-query-images").exists())


if __name__ == "__main__":
    unittest.main()
