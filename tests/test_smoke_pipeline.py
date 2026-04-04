import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
import json
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

    def test_cli_prints_secmi_plan(self) -> None:
        from diffaudit.cli import main

        config_text = """
task:
  name: secmi-plan
  model_family: diffusion
  access_level: black_box
assets:
  dataset_id: cifar10-half
  dataset_name: cifar10
  dataset_root: D:/datasets/cifar10
  model_id: cifar10-ddpm
  model_dir: D:/models/secmi/cifar10
attack:
  method: secmi
  num_samples: 8
  parameters:
    t_sec: 100
    k: 10
report:
  output_dir: experiments/secmi-plan
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "audit.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["plan-secmi", "--config", str(config_path)])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["entrypoint"], "mia_evals/secmia.py")
        self.assertEqual(payload["dataset"], "cifar10")

    def test_builds_secmi_plan_from_config(self) -> None:
        from diffaudit.attacks.secmi import build_secmi_plan
        from diffaudit.config import load_audit_config

        config_text = """
task:
  name: secmi-plan
  model_family: diffusion
  access_level: black_box
assets:
  dataset_id: cifar10-half
  dataset_name: cifar10
  dataset_root: D:/datasets/cifar10
  model_id: cifar10-ddpm
  model_dir: D:/models/secmi/cifar10
attack:
  method: secmi
  num_samples: 8
  parameters:
    t_sec: 100
    k: 10
    batch_size: 16
report:
  output_dir: experiments/secmi-plan
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "audit.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            config = load_audit_config(config_path)
            plan = build_secmi_plan(config)

        self.assertEqual(plan.entrypoint, "mia_evals/secmia.py")
        self.assertEqual(plan.dataset, "cifar10")
        self.assertEqual(plan.data_root, "D:/datasets/cifar10")
        self.assertEqual(plan.model_dir, "D:/models/secmi/cifar10")
        self.assertEqual(plan.t_sec, 100)
        self.assertEqual(plan.k, 10)
        self.assertEqual(plan.batch_size, 16)

    def test_resolves_secmi_artifacts_from_model_dir(self) -> None:
        from diffaudit.attacks.secmi import resolve_secmi_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "flagfile.txt").write_text("flags", encoding="utf-8")
            (model_dir / "ckpt-step100.pt").write_text("100", encoding="utf-8")
            (model_dir / "ckpt-step200.pt").write_text("200", encoding="utf-8")

            artifacts = resolve_secmi_artifacts(model_dir)

        self.assertTrue(artifacts.checkpoint_path.endswith("ckpt-step200.pt"))
        self.assertTrue(artifacts.flagfile_path.endswith("flagfile.txt"))

    def test_prefers_checkpoint_pt_when_present(self) -> None:
        from diffaudit.attacks.secmi import resolve_secmi_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "flagfile.txt").write_text("flags", encoding="utf-8")
            (model_dir / "checkpoint.pt").write_text("best", encoding="utf-8")
            (model_dir / "ckpt-step200.pt").write_text("200", encoding="utf-8")

            artifacts = resolve_secmi_artifacts(model_dir)

        self.assertTrue(artifacts.checkpoint_path.endswith("checkpoint.pt"))


if __name__ == "__main__":
    unittest.main()
