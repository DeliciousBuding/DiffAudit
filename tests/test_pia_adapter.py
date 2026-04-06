import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


PIA_CONFIG_TEMPLATE = """
task:
  name: pia-runtime
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
    attack_num: 3
    interval: 10
report:
  output_dir: experiments/pia-runtime
"""


def create_minimal_pia_repo(repo_root: Path) -> Path:
    ddpm_root = repo_root / "DDPM"
    ddpm_root.mkdir(parents=True)
    (ddpm_root / "attack.py").write_text("# attack entrypoint\n", encoding="utf-8")
    (ddpm_root / "dataset_utils.py").write_text("# dataset utils\n", encoding="utf-8")
    (ddpm_root / "components.py").write_text(
        """
import torch


class EpsGetter:
    def __init__(self, model):
        self.model = model


class PIA:
    def __init__(self, betas, interval, attack_num, eps_getter, normalize=None, denormalize=None, lp=4):
        self.betas = betas
        self.interval = interval
        self.attack_num = attack_num
        self.eps_getter = eps_getter
        self.normalize = normalize
        self.lp = lp

    def __call__(self, x0, condition=None):
        if self.normalize is not None:
            x0 = self.normalize(x0)
        eps = self.eps_getter(x0, condition, None, 0)
        base = eps.flatten(1).abs().mean(dim=1)
        return torch.stack([base + float(i) for i in range(self.attack_num)], dim=0)
""".strip(),
        encoding="utf-8",
    )
    (ddpm_root / "model.py").write_text(
        """
import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x, t):
        del t
        return x * self.scale
""".strip(),
        encoding="utf-8",
    )
    return ddpm_root


class PiaAdapterTests(unittest.TestCase):
    def test_bootstrap_pia_smoke_assets_creates_checkpoint(self) -> None:
        from diffaudit.attacks.pia_adapter import bootstrap_pia_smoke_assets

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "PIA"
            create_minimal_pia_repo(repo_root)
            target_dir = root / "model"

            result = bootstrap_pia_smoke_assets(target_dir=target_dir, repo_root=repo_root)

        self.assertTrue(result["checkpoint_path"].endswith("checkpoint.pt"))
        self.assertEqual(result["weights_key"], "ema_model")

    def test_cli_runtime_probe_pia_reports_ready(self) -> None:
        from diffaudit.attacks.pia_adapter import bootstrap_pia_smoke_assets
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "PIA"
            create_minimal_pia_repo(repo_root)

            dataset_root = root / "datasets"
            (dataset_root / "cifar10").mkdir(parents=True)

            member_split_root = root / "member_splits"
            member_split_root.mkdir()
            (member_split_root / "CIFAR10_train_ratio0.5.npz").write_bytes(b"split")

            model_dir = root / "model"
            bootstrap_pia_smoke_assets(target_dir=model_dir, repo_root=repo_root)

            config_path = root / "audit.yaml"
            config_path.write_text(
                PIA_CONFIG_TEMPLATE
                .replace("PLACEHOLDER_DATASET", str(dataset_root).replace("\\", "/"))
                .replace("PLACEHOLDER_MODEL", str(model_dir).replace("\\", "/")),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "runtime-probe-pia",
                        "--config",
                        str(config_path),
                        "--repo-root",
                        str(repo_root),
                        "--member-split-root",
                        str(member_split_root),
                        "--device",
                        "cpu",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["checks"]["model_loaded"])
        self.assertTrue(payload["checks"]["attacker_instantiated"])

    def test_cli_runtime_preview_pia_reports_ready(self) -> None:
        from PIL import Image
        import numpy as np

        from diffaudit.attacks.pia_adapter import bootstrap_pia_smoke_assets
        from diffaudit.cli import main

        class FakeCIFAR10:
            def __init__(self, root, train, transform=None, download=False):
                del root, train, download
                self.transform = transform
                self.images = [
                    np.full((32, 32, 3), fill_value=value, dtype=np.uint8)
                    for value in (32, 64, 160, 192)
                ]

            def __len__(self) -> int:
                return len(self.images)

            def __getitem__(self, index: int):
                image = Image.fromarray(self.images[index])
                tensor = self.transform(image) if self.transform is not None else image
                return tensor, index

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "PIA"
            create_minimal_pia_repo(repo_root)

            dataset_root = root / "datasets"
            (dataset_root / "cifar10").mkdir(parents=True)

            member_split_root = root / "member_splits"
            member_split_root.mkdir()
            np.savez(
                member_split_root / "CIFAR10_train_ratio0.5.npz",
                mia_train_idxs=np.array([0, 1]),
                mia_eval_idxs=np.array([2, 3]),
            )

            model_dir = root / "model"
            bootstrap_pia_smoke_assets(target_dir=model_dir, repo_root=repo_root)

            config_path = root / "audit.yaml"
            config_path.write_text(
                PIA_CONFIG_TEMPLATE
                .replace("PLACEHOLDER_DATASET", str(dataset_root).replace("\\", "/"))
                .replace("PLACEHOLDER_MODEL", str(model_dir).replace("\\", "/")),
                encoding="utf-8",
            )

            stdout = StringIO()
            with patch("torchvision.datasets.CIFAR10", FakeCIFAR10):
                with redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "runtime-preview-pia",
                            "--config",
                            str(config_path),
                            "--repo-root",
                            str(repo_root),
                            "--member-split-root",
                            str(member_split_root),
                            "--device",
                            "cpu",
                            "--preview-batch-size",
                            "2",
                        ]
                    )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["member_batch_shape"], [2, 3, 32, 32])
        self.assertEqual(payload["nonmember_batch_shape"], [2, 3, 32, 32])
        self.assertEqual(payload["member_score_shape"], [3, 2])
        self.assertEqual(payload["nonmember_score_shape"], [3, 2])
        self.assertTrue(payload["checks"]["member_preview_loaded"])
        self.assertTrue(payload["checks"]["nonmember_preview_loaded"])

    def test_cli_runs_pia_synthetic_smoke(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "PIA"
            create_minimal_pia_repo(repo_root)

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "run-pia-synth-smoke",
                        "--workspace",
                        str(root / "pia-synth-smoke"),
                        "--repo-root",
                        str(repo_root),
                        "--device",
                        "cpu",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["method"], "pia")
        self.assertEqual(payload["mode"], "synthetic-smoke")
        self.assertEqual(payload["device"], "cpu")
        self.assertIn("auc", payload["metrics"])
        self.assertIn("summary", payload["artifact_paths"])
        self.assertFalse((root / "pia-synth-smoke" / "synthetic-pia-assets").exists())

    def test_run_pia_runtime_smoke_writes_summary(self) -> None:
        from diffaudit.attacks.pia_adapter import run_pia_runtime_smoke

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "PIA"
            create_minimal_pia_repo(repo_root)

            result = run_pia_runtime_smoke(
                workspace=root / "pia-runtime-smoke",
                repo_root=repo_root,
                device="cpu",
            )

            self.assertTrue((root / "pia-runtime-smoke" / "summary.json").exists())
            self.assertFalse((root / "pia-runtime-smoke" / "synthetic-assets").exists())

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["mode"], "runtime-smoke")
        self.assertEqual(result["device"], "cpu")

    def test_cli_runs_pia_runtime_smoke(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "PIA"
            create_minimal_pia_repo(repo_root)

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "run-pia-runtime-smoke",
                        "--workspace",
                        str(root / "pia-runtime-smoke"),
                        "--repo-root",
                        str(repo_root),
                        "--device",
                        "cpu",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["mode"], "runtime-smoke")
        self.assertTrue(payload["checks"]["runtime_probe_ready"])


if __name__ == "__main__":
    unittest.main()
