import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


class SmokePipelineTests(unittest.TestCase):
    def test_loads_audit_config_and_builds_summary(self) -> None:
        from diffaudit.config import load_audit_config
        from diffaudit.pipelines.smoke import build_smoke_summary

        config_text = """
task:
  name: secmi-smoke
  model_family: diffusion
  access_level: black_box
assets:
  dataset_id: toy-membership-set
  model_id: stable-diffusion-smoke
attack:
  method: secmi
  num_samples: 8
report:
  output_dir: experiments/secmi-smoke
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "audit.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            config = load_audit_config(config_path)
            summary = build_smoke_summary(config)

        self.assertEqual(config.task.name, "secmi-smoke")
        self.assertEqual(config.attack.method, "secmi")
        self.assertEqual(summary["run_name"], "secmi-smoke")
        self.assertEqual(summary["dataset_id"], "toy-membership-set")
        self.assertEqual(summary["num_samples"], 8)
        self.assertEqual(summary["output_dir"], "experiments/secmi-smoke")

    def test_rejects_unsupported_access_level(self) -> None:
        from diffaudit.config import load_audit_config

        config_text = """
task:
  name: bad-config
  model_family: diffusion
  access_level: internet
assets:
  dataset_id: toy-membership-set
  model_id: stable-diffusion-smoke
attack:
  method: secmi
  num_samples: 8
report:
  output_dir: experiments/bad-config
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "audit.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsupported access_level"):
                load_audit_config(config_path)

    def test_cli_runs_smoke_and_writes_summary_file(self) -> None:
        from diffaudit.cli import main

        config_text = """
task:
  name: cli-smoke
  model_family: diffusion
  access_level: black_box
assets:
  dataset_id: toy-membership-set
  model_id: stable-diffusion-smoke
attack:
  method: secmi
  num_samples: 4
report:
  output_dir: experiments/cli-smoke
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            config_path = workspace / "audit.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "run-smoke",
                        "--config",
                        str(config_path),
                        "--workspace",
                        str(workspace),
                    ]
                )

            summary_path = workspace / "experiments" / "cli-smoke" / "summary.json"

            self.assertEqual(exit_code, 0)
            self.assertTrue(summary_path.exists())
            self.assertIn("summary.json", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
