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

    def test_resolves_secmi_artifacts_without_flagfile(self) -> None:
        from diffaudit.attacks.secmi import resolve_secmi_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "checkpoint.pt").write_text("best", encoding="utf-8")

            artifacts = resolve_secmi_artifacts(model_dir)

        self.assertTrue(artifacts.checkpoint_path.endswith("checkpoint.pt"))
        self.assertIsNone(artifacts.flagfile_path)

    def test_builds_secmi_runner_spec_from_local_vendor_repo(self) -> None:
        from diffaudit.attacks.secmi import (
            build_secmi_plan,
            build_secmi_runner_spec,
            resolve_secmi_artifacts,
        )
        from diffaudit.config import load_audit_config

        config_text = """
task:
  name: secmi-runner
  model_family: diffusion
  access_level: black_box
assets:
  dataset_id: cifar10-half
  dataset_name: cifar10
  dataset_root: D:/datasets/cifar10
  model_id: cifar10-ddpm
  model_dir: PLACEHOLDER
attack:
  method: secmi
  num_samples: 8
  parameters:
    t_sec: 100
    k: 10
report:
  output_dir: experiments/secmi-runner
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            model_dir = workspace / "model"
            model_dir.mkdir()
            (model_dir / "flagfile.txt").write_text("flags", encoding="utf-8")
            (model_dir / "checkpoint.pt").write_text("best", encoding="utf-8")

            repo_root = workspace / "SecMI"
            (repo_root / "mia_evals").mkdir(parents=True)
            (repo_root / "mia_evals" / "secmia.py").write_text("# secmi", encoding="utf-8")

            config_path = workspace / "audit.yaml"
            config_path.write_text(
                config_text.replace("PLACEHOLDER", str(model_dir).replace("\\", "/")),
                encoding="utf-8",
            )

            config = load_audit_config(config_path)
            plan = build_secmi_plan(config)
            artifacts = resolve_secmi_artifacts(model_dir)
            spec = build_secmi_runner_spec(plan, artifacts, repo_root)

        self.assertTrue(spec.entrypoint_path.endswith("mia_evals\\secmia.py"))
        self.assertEqual(spec.python_module, "mia_evals.secmia")
        self.assertIn("checkpoint.pt", spec.checkpoint_path)

    def test_validates_secmi_workspace_layout(self) -> None:
        from diffaudit.attacks.secmi import validate_secmi_workspace

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "SecMI"
            (workspace / "mia_evals").mkdir(parents=True)
            (workspace / "mia_evals" / "secmia.py").write_text("print('ok')", encoding="utf-8")
            (workspace / "mia_evals" / "dataset_utils.py").write_text("# util", encoding="utf-8")
            (workspace / "model.py").write_text("# model", encoding="utf-8")
            (workspace / "diffusion.py").write_text("# diffusion", encoding="utf-8")

            result = validate_secmi_workspace(workspace)

        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["entrypoint"].endswith("mia_evals\\secmia.py"))

class SecmiAssetExplanationTests(unittest.TestCase):
    def test_explains_missing_secmi_assets(self) -> None:
        from diffaudit.attacks.secmi import explain_secmi_assets
        from diffaudit.config import load_audit_config

        config_text = """
task:
  name: secmi-assets
  model_family: diffusion
  access_level: black_box
assets:
  dataset_id: cifar10-half
  dataset_name: cifar10
  dataset_root: D:/datasets/cifar10
  model_id: cifar10-ddpm
  model_dir: D:/missing/secmi-model
attack:
  method: secmi
  num_samples: 8
  parameters:
    t_sec: 100
    k: 10
report:
  output_dir: experiments/secmi-assets
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "audit.yaml"
            config_path.write_text(config_text, encoding="utf-8")
            config = load_audit_config(config_path)
            summary = explain_secmi_assets(config)

        self.assertEqual(summary["status"], "blocked")
        self.assertIn("checkpoint", summary["missing_description"])


if __name__ == "__main__":
    unittest.main()

