import unittest
from pathlib import Path
import tempfile
from contextlib import redirect_stdout
from io import StringIO
import json


class AttackRegistryTests(unittest.TestCase):
    def test_returns_secmi_planner(self) -> None:
        from diffaudit.attacks.registry import get_attack_planner

        planner = get_attack_planner("secmi")

        self.assertEqual(planner.__name__, "build_secmi_plan")

    def test_rejects_unknown_attack(self) -> None:
        from diffaudit.attacks.registry import get_attack_planner

        with self.assertRaisesRegex(ValueError, "Unsupported attack method"):
            get_attack_planner("unknown")

    def test_vendored_secmi_module_imports(self) -> None:
        from third_party.secmi.mia_evals import secmia

        self.assertTrue(hasattr(secmia, "get_FLAGS"))
        self.assertTrue(hasattr(secmia, "secmi_attack"))

    def test_validates_external_secmi_workspace(self) -> None:
        from diffaudit.attacks.secmi import validate_secmi_workspace

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "SecMI"
            (workspace / "mia_evals").mkdir(parents=True)
            (workspace / "mia_evals" / "secmia.py").write_text("# secmi", encoding="utf-8")
            (workspace / "mia_evals" / "dataset_utils.py").write_text("# util", encoding="utf-8")
            (workspace / "model.py").write_text("# model", encoding="utf-8")
            (workspace / "diffusion.py").write_text("# diffusion", encoding="utf-8")

            result = validate_secmi_workspace(workspace)

        self.assertEqual(result["status"], "ready")
        self.assertIn("mia_evals", result["entrypoint"])

    def test_cli_prepares_secmi_adapter(self) -> None:
        from diffaudit.cli import main

        config_text = """
task:
  name: secmi-adapter
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
  output_dir: experiments/secmi-adapter
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "flagfile.txt").write_text("flags", encoding="utf-8")
            (model_dir / "checkpoint.pt").write_text("best", encoding="utf-8")

            repo_root = root / "third_party" / "secmi"
            (repo_root / "mia_evals").mkdir(parents=True)
            (repo_root / "__init__.py").write_text('"""secmi"""', encoding="utf-8")
            (repo_root / "model.py").write_text("# model", encoding="utf-8")
            (repo_root / "diffusion.py").write_text("# diffusion", encoding="utf-8")
            (repo_root / "mia_evals" / "__init__.py").write_text('"""mia"""', encoding="utf-8")
            (repo_root / "mia_evals" / "dataset_utils.py").write_text("# util", encoding="utf-8")
            (repo_root / "mia_evals" / "secmia.py").write_text(
                "def get_FLAGS(*args, **kwargs):\n    return None\n"
                "def secmi_attack(*args, **kwargs):\n    return None\n",
                encoding="utf-8",
            )

            config_path = root / "audit.yaml"
            config_path.write_text(
                config_text.replace("PLACEHOLDER", str(model_dir).replace("\\", "/")),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "prepare-secmi",
                        "--config",
                        str(config_path),
                        "--repo-root",
                        str(repo_root),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["python_module"], "mia_evals.secmia")
        self.assertIn("checkpoint.pt", payload["checkpoint_path"])

    def test_cli_runs_secmi_dry_run(self) -> None:
        from diffaudit.cli import main

        config_text = """
task:
  name: secmi-dry-run
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
  output_dir: experiments/secmi-dry-run
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "flagfile.txt").write_text("", encoding="utf-8")
            (model_dir / "checkpoint.pt").write_text("best", encoding="utf-8")

            config_path = root / "audit.yaml"
            config_path.write_text(
                config_text.replace("PLACEHOLDER", str(model_dir).replace("\\", "/")),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "dry-run-secmi",
                        "--config",
                        str(config_path),
                        "--repo-root",
                        "external/SecMI",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertIn("get_FLAGS", payload["available_functions"])

    def test_cli_dry_run_reports_missing_assets(self) -> None:
        from diffaudit.cli import main

        config_text = """
task:
  name: secmi-dry-run
  model_family: diffusion
  access_level: black_box
assets:
  dataset_id: cifar10-half
  dataset_name: cifar10
  dataset_root: D:/datasets/cifar10
  model_id: cifar10-ddpm
  model_dir: D:/missing/model
attack:
  method: secmi
  num_samples: 8
  parameters:
    t_sec: 100
    k: 10
report:
  output_dir: experiments/secmi-dry-run
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "audit.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "dry-run-secmi",
                        "--config",
                        str(config_path),
                        "--repo-root",
                        "external/SecMI",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["status"], "blocked")
        self.assertIn("flagfile", payload["error"])

    def test_cli_probes_secmi_assets(self) -> None:
        from diffaudit.cli import main

        config_text = """
task:
  name: secmi-asset-probe
  model_family: diffusion
  access_level: black_box
assets:
  dataset_id: cifar10-half
  dataset_name: cifar10
  dataset_root: PLACEHOLDER_DATASET
  model_id: cifar10-ddpm
  model_dir: PLACEHOLDER_MODEL
attack:
  method: secmi
  num_samples: 8
  parameters:
    t_sec: 100
    k: 10
report:
  output_dir: experiments/secmi-asset-probe
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            dataset_root.mkdir()

            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "flagfile.txt").write_text("flags", encoding="utf-8")
            (model_dir / "ckpt-step100.pt").write_text("checkpoint", encoding="utf-8")

            member_split_root = root / "member_splits"
            member_split_root.mkdir()
            (member_split_root / "CIFAR10_train_ratio0.5.npz").write_bytes(b"split")

            config_path = root / "audit.yaml"
            config_path.write_text(
                config_text
                .replace("PLACEHOLDER_DATASET", str(dataset_root).replace("\\", "/"))
                .replace("PLACEHOLDER_MODEL", str(model_dir).replace("\\", "/")),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "probe-secmi-assets",
                        "--config",
                        str(config_path),
                        "--member-split-root",
                        str(member_split_root),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["checks"]["checkpoint"])
        self.assertTrue(payload["checks"]["member_split"])

    def test_cli_probe_secmi_assets_reports_missing_items(self) -> None:
        from diffaudit.cli import main

        config_text = """
task:
  name: secmi-asset-probe
  model_family: diffusion
  access_level: black_box
assets:
  dataset_id: cifar10-half
  dataset_name: cifar10
  dataset_root: D:/missing/dataset-root
  model_id: cifar10-ddpm
  model_dir: D:/missing/model-root
attack:
  method: secmi
  num_samples: 8
  parameters:
    t_sec: 100
    k: 10
report:
  output_dir: experiments/secmi-asset-probe
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            member_split_root = root / "member_splits"
            member_split_root.mkdir()
            config_path = root / "audit.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "probe-secmi-assets",
                        "--config",
                        str(config_path),
                        "--member-split-root",
                        str(member_split_root),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["status"], "blocked")
        self.assertIn("flagfile.txt", payload["missing_items"])
        self.assertIn("checkpoint", payload["missing_description"])

    def test_parses_secmi_flagfile_without_absl_state(self) -> None:
        from diffaudit.attacks.secmi_adapter import parse_secmi_flagfile

        flag_text = """--T=100
--attn=1
--batch_size=8
--beta_1=0.0001
--beta_T=0.02
--ch=32
--ch_mult=1
--ch_mult=2
--dropout=0.1
--img_size=32
--num_res_blocks=1
--mean_type=epsilon
--var_type=fixedlarge
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            flagfile = Path(tmpdir) / "flagfile.txt"
            flagfile.write_text(flag_text, encoding="utf-8")
            flags = parse_secmi_flagfile(flagfile)

        self.assertEqual(flags.T, 100)
        self.assertEqual(flags.ch, 32)
        self.assertEqual(flags.ch_mult, [1, 2])
        self.assertEqual(flags.num_res_blocks, 1)

    def test_runs_synthetic_secmi_stat_smoke(self) -> None:
        from diffaudit.attacks.secmi_adapter import run_synthetic_secmi_stat_smoke

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            result = run_synthetic_secmi_stat_smoke(workspace, device="cpu")
            self.assertEqual(result["status"], "ready")
            self.assertIn("auc", result)
            self.assertIn("asr", result)
            self.assertEqual(result["device"], "cpu")
            self.assertTrue((workspace / "summary.json").exists())
            self.assertFalse((workspace / "synthetic-secmi-assets").exists())

    def test_bootstrap_secmi_smoke_assets_creates_flagfile_and_checkpoint(self) -> None:
        from diffaudit.attacks.secmi_adapter import bootstrap_secmi_smoke_assets

        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir) / "smoke-assets"
            result = bootstrap_secmi_smoke_assets(
                target_dir=target_dir,
                flagfile_source=Path("external/SecMI/config/CIFAR10.txt"),
            )

        self.assertTrue(result["flagfile_path"].endswith("flagfile.txt"))
        self.assertTrue(result["checkpoint_path"].endswith("checkpoint.pt"))

    def test_cli_runtime_probe_secmi_reports_ready(self) -> None:
        from diffaudit.cli import main
        from diffaudit.attacks.secmi_adapter import bootstrap_secmi_smoke_assets

        config_text = """
task:
  name: secmi-runtime
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
  output_dir: experiments/secmi-runtime
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            bootstrap_secmi_smoke_assets(
                target_dir=model_dir,
                flagfile_source=Path("external/SecMI/config/CIFAR10.txt"),
            )

            config_path = root / "audit.yaml"
            config_path.write_text(
                config_text.replace("PLACEHOLDER", str(model_dir).replace("\\", "/")),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "runtime-probe-secmi",
                        "--config",
                        str(config_path),
                        "--repo-root",
                        "third_party/secmi",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertIn("flagfile_loaded", payload["checks"])

    def test_cli_runs_secmi_synthetic_smoke(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "run-secmi-synth-smoke",
                        "--workspace",
                        tmpdir,
                        "--device",
                        "cpu",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["device"], "cpu")
        self.assertIn("auc", payload)

    def test_prepares_secmi_adapter_context(self) -> None:
        from diffaudit.attacks.secmi_adapter import prepare_secmi_adapter
        from diffaudit.config import load_audit_config

        config_text = """
task:
  name: secmi-adapter
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
  output_dir: experiments/secmi-adapter
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "flagfile.txt").write_text("flags", encoding="utf-8")
            (model_dir / "checkpoint.pt").write_text("best", encoding="utf-8")

            config_path = Path(tmpdir) / "audit.yaml"
            config_path.write_text(
                config_text.replace("PLACEHOLDER", str(model_dir).replace("\\", "/")),
                encoding="utf-8",
            )

            config = load_audit_config(config_path)
            context = prepare_secmi_adapter(config, "external/SecMI")

        self.assertEqual(context.plan.dataset, "cifar10")
        self.assertTrue(context.runner.entrypoint_path.endswith("mia_evals\\secmia.py"))
        self.assertEqual(context.module.__name__, "third_party.secmi.mia_evals.secmia")


if __name__ == "__main__":
    unittest.main()

