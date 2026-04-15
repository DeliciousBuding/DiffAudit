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

    def test_returns_pia_planner(self) -> None:
        from diffaudit.attacks.registry import get_attack_planner

        planner = get_attack_planner("pia")

        self.assertEqual(planner.__name__, "build_pia_plan")

    def test_returns_clid_planner(self) -> None:
        from diffaudit.attacks.registry import get_attack_planner

        planner = get_attack_planner("clid")

        self.assertEqual(planner.__name__, "build_clid_plan")

    def test_returns_recon_planner(self) -> None:
        from diffaudit.attacks.registry import get_attack_planner

        planner = get_attack_planner("recon")

        self.assertEqual(planner.__name__, "build_recon_plan")

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
        self.assertIn("checkpoint", payload["error"])

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
        self.assertIn("checkpoint", payload["missing_description"])

    def test_cli_probe_secmi_assets_allows_missing_flagfile(self) -> None:
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
            (model_dir / "checkpoint.pt").write_text("checkpoint", encoding="utf-8")

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
        self.assertFalse(payload["checks"]["flagfile"])
        self.assertTrue(payload["checks"]["checkpoint"])

    def test_cli_plans_pia(self) -> None:
        from diffaudit.cli import main

        config_text = """
task:
  name: pia-plan
  model_family: diffusion
  access_level: semi_white_box
assets:
  dataset_id: cifar10-half
  dataset_name: cifar10
  dataset_root: D:/datasets
  model_id: cifar10-ddpm
  model_dir: D:/checkpoints/pia
attack:
  method: pia
  num_samples: 8
  parameters:
    attacker_name: PIA
    attack_num: 30
    interval: 10
report:
  output_dir: experiments/pia-plan
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "audit.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["plan-pia", "--config", str(config_path)])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["entrypoint"], "DDPM/attack.py")
        self.assertEqual(payload["attack_num"], 30)
        self.assertEqual(payload["interval"], 10)

    def test_cli_probes_pia_assets(self) -> None:
        from diffaudit.cli import main

        config_text = """
task:
  name: pia-probe
  model_family: diffusion
  access_level: semi_white_box
assets:
  dataset_id: cifar10-half
  dataset_name: cifar10
  dataset_root: PLACEHOLDER_DATASET
  model_id: cifar10-ddpm
  model_dir: PLACEHOLDER_MODEL
attack:
  method: pia
  num_samples: 8
  parameters:
    attacker_name: PIA
    attack_num: 30
    interval: 10
report:
  output_dir: experiments/pia-probe
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "data-root"
            (dataset_root / "cifar10").mkdir(parents=True)

            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "checkpoint.pt").write_text("checkpoint", encoding="utf-8")

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
                        "probe-pia-assets",
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

    def test_cli_runs_pia_dry_run(self) -> None:
        from diffaudit.cli import main

        config_text = """
task:
  name: pia-dry-run
  model_family: diffusion
  access_level: semi_white_box
assets:
  dataset_id: cifar10-half
  dataset_name: cifar10
  dataset_root: PLACEHOLDER_DATASET
  model_id: cifar10-ddpm
  model_dir: PLACEHOLDER_MODEL
attack:
  method: pia
  num_samples: 8
  parameters:
    attacker_name: PIA
    attack_num: 30
    interval: 10
report:
  output_dir: experiments/pia-dry-run
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "data-root"
            (dataset_root / "cifar10").mkdir(parents=True)

            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "checkpoint.pt").write_text("checkpoint", encoding="utf-8")

            member_split_root = root / "member_splits"
            member_split_root.mkdir()
            (member_split_root / "CIFAR10_train_ratio0.5.npz").write_bytes(b"split")

            repo_root = root / "PIA"
            ddpm_root = repo_root / "DDPM"
            ddpm_root.mkdir(parents=True)
            (ddpm_root / "attack.py").write_text("# attack", encoding="utf-8")
            (ddpm_root / "components.py").write_text("class PIA:\n    pass\n", encoding="utf-8")
            (ddpm_root / "dataset_utils.py").write_text("# dataset utils", encoding="utf-8")
            (ddpm_root / "model.py").write_text("class UNet:\n    pass\n", encoding="utf-8")

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
                        "dry-run-pia",
                        "--config",
                        str(config_path),
                        "--repo-root",
                        str(repo_root),
                        "--member-split-root",
                        str(member_split_root),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["checks"]["workspace_files"])
        self.assertTrue(payload["checks"]["components_has_pia"])
        self.assertTrue(payload["checks"]["model_has_unet"])

    def test_cli_plans_clid(self) -> None:
        from diffaudit.cli import main

        config_text = """
task:
  name: clid-plan
  model_family: diffusion
  access_level: black_box
assets:
  dataset_id: coco-split1
  dataset_train_ref: zsf/COCO_MIA_ori_split1
  dataset_test_ref: zsf/COCO_MIA_ori_split1
  model_id: sd15-shadow
  model_dir: D:/models/clid/sd15
attack:
  method: clid
  num_samples: 8
  parameters:
    variant: clid_impt
    max_n_samples: 3
report:
  output_dir: experiments/clid-plan
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "audit.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["plan-clid", "--config", str(config_path)])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["entrypoint"], "mia_CLiD_impt.py")
        self.assertEqual(payload["attack_variant"], "clid_impt")
        self.assertEqual(payload["max_n_samples"], 3)

    def test_cli_probes_clid_assets(self) -> None:
        from diffaudit.cli import main

        config_text = """
task:
  name: clid-probe
  model_family: diffusion
  access_level: black_box
assets:
  dataset_id: coco-split1
  dataset_train_ref: zsf/COCO_MIA_ori_split1
  dataset_test_ref: zsf/COCO_MIA_ori_split1
  model_id: sd15-shadow
  model_dir: PLACEHOLDER_MODEL
attack:
  method: clid
  num_samples: 8
  parameters:
    variant: clid_impt
    max_n_samples: 3
report:
  output_dir: experiments/clid-probe
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            for subdir in ("vae", "tokenizer", "text_encoder", "unet", "scheduler"):
                (model_dir / subdir).mkdir(parents=True)

            config_path = root / "audit.yaml"
            config_path.write_text(
                config_text.replace("PLACEHOLDER_MODEL", str(model_dir).replace("\\", "/")),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["probe-clid-assets", "--config", str(config_path)])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["checks"]["model_subdirs"])
        self.assertTrue(payload["checks"]["dataset_train_ref"])

    def test_cli_runs_clid_dry_run(self) -> None:
        from diffaudit.cli import main

        config_text = """
task:
  name: clid-dry-run
  model_family: diffusion
  access_level: black_box
assets:
  dataset_id: coco-split1
  dataset_train_ref: zsf/COCO_MIA_ori_split1
  dataset_test_ref: zsf/COCO_MIA_ori_split1
  model_id: sd15-shadow
  model_dir: PLACEHOLDER_MODEL
attack:
  method: clid
  num_samples: 8
  parameters:
    variant: clid_impt
    max_n_samples: 3
report:
  output_dir: experiments/clid-dry-run
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            for subdir in ("vae", "tokenizer", "text_encoder", "unet", "scheduler"):
                (model_dir / subdir).mkdir(parents=True)

            repo_root = root / "CLiD"
            repo_root.mkdir()
            (repo_root / "mia_CLiD_impt.py").write_text(
                "from datasets import load_dataset\n"
                "flags.attack = 'clid_impt'\n"
                "from diffusers import AutoencoderKL\n",
                encoding="utf-8",
            )
            (repo_root / "cal_clid_th.py").write_text("# cal", encoding="utf-8")
            (repo_root / "cal_clid_xgb.py").write_text("# cal", encoding="utf-8")
            (repo_root / "train_text_to_image.py").write_text("# train", encoding="utf-8")

            config_path = root / "audit.yaml"
            config_path.write_text(
                config_text.replace("PLACEHOLDER_MODEL", str(model_dir).replace("\\", "/")),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "dry-run-clid",
                        "--config",
                        str(config_path),
                        "--repo-root",
                        str(repo_root),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["checks"]["entrypoint_has_attack_flag"])
        self.assertTrue(payload["checks"]["script_uses_load_dataset"])

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

