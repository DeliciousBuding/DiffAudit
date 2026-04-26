import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image


class TmiadmAdapterTests(unittest.TestCase):
    def test_compute_tmiadm_family_scores_returns_expected_families(self) -> None:
        from diffaudit.attacks.tmiadm_adapter import _compute_tmiadm_family_scores

        timestep_predictions = torch.tensor(
            [
                [
                    [[[0.2, 0.2], [0.2, 0.2]]],
                    [[[0.8, 0.8], [0.8, 0.8]]],
                    [[[1.4, 1.4], [1.4, 1.4]]],
                ],
                [
                    [[[0.3, 0.3], [0.3, 0.3]]],
                    [[[0.35, 0.35], [0.35, 0.35]]],
                    [[[0.4, 0.4], [0.4, 0.4]]],
                ],
            ],
            dtype=torch.float32,
        )

        scores = _compute_tmiadm_family_scores(timestep_predictions, aggregation_p_norm=2)

        self.assertEqual(sorted(scores.keys()), ["fused", "long_window", "short_window"])
        self.assertEqual(list(scores["short_window"].shape), [2])
        self.assertEqual(list(scores["long_window"].shape), [2])
        self.assertEqual(list(scores["fused"].shape), [2])
        self.assertGreater(scores["short_window"][1].item(), scores["short_window"][0].item())

    def test_protocol_probe_runs_on_minimal_pia_assets(self) -> None:
        from diffaudit.attacks.pia_adapter import bootstrap_pia_smoke_assets
        from diffaudit.attacks.tmiadm_adapter import run_tmiadm_protocol_probe
        from diffaudit.config import load_audit_config
        from tests.test_pia_adapter import PIA_CONFIG_TEMPLATE, create_minimal_pia_repo

        class FakeCIFAR10:
            def __init__(self, root, train, transform=None, download=False):
                del root, train, download
                self.transform = transform
                self.images = [
                    np.full((32, 32, 3), fill_value=value, dtype=np.uint8)
                    for value in (24, 56, 152, 216)
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
            config = load_audit_config(config_path)

            with patch("diffaudit.attacks.pia_adapter.tv_datasets.CIFAR10", FakeCIFAR10):
                result = run_tmiadm_protocol_probe(
                    config=config,
                    workspace=root / "tmiadm-protocol-probe",
                    repo_root=repo_root,
                    member_split_root=member_split_root,
                    device="cpu",
                    max_samples=2,
                    batch_size=2,
                    scan_timesteps=[10, 20, 30],
                    noise_seed=0,
                )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["method"], "tmiadm")
        self.assertEqual(result["mode"], "protocol-probe")
        self.assertEqual(sorted(result["family_metrics"].keys()), ["fused", "long_window", "short_window"])
        self.assertIn(result["best_family"], ["fused", "long_window", "short_window"])

    def test_protocol_probe_accepts_stochastic_dropout_defense(self) -> None:
        from diffaudit.attacks.pia_adapter import bootstrap_pia_smoke_assets
        from diffaudit.attacks.tmiadm_adapter import run_tmiadm_protocol_probe
        from diffaudit.config import load_audit_config
        from tests.test_pia_adapter import PIA_CONFIG_TEMPLATE, create_minimal_pia_repo

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
            config = load_audit_config(config_path)

            with patch("diffaudit.attacks.pia_adapter.tv_datasets.CIFAR10", FakeCIFAR10):
                result = run_tmiadm_protocol_probe(
                    config=config,
                    workspace=root / "tmiadm-protocol-probe-defense",
                    repo_root=repo_root,
                    member_split_root=member_split_root,
                    device="cpu",
                    max_samples=2,
                    batch_size=2,
                    scan_timesteps=[10, 20, 30],
                    noise_seed=0,
                    stochastic_dropout_defense=True,
                    dropout_activation_schedule="all_steps",
                )

        self.assertTrue(result["defense"]["enabled"])
        self.assertEqual(result["defense"]["name"], "stochastic-dropout")
        self.assertEqual(result["defense"]["dropout_activation_schedule"], "all_steps")

    def test_protocol_probe_accepts_timestep_jitter_defense(self) -> None:
        from diffaudit.attacks.pia_adapter import bootstrap_pia_smoke_assets
        from diffaudit.attacks.tmiadm_adapter import run_tmiadm_protocol_probe
        from diffaudit.config import load_audit_config
        from tests.test_pia_adapter import PIA_CONFIG_TEMPLATE, create_minimal_pia_repo

        class FakeCIFAR10:
            def __init__(self, root, train, transform=None, download=False):
                del root, train, download
                self.transform = transform
                self.images = [
                    np.full((32, 32, 3), fill_value=value, dtype=np.uint8)
                    for value in (40, 72, 168, 200)
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
            config = load_audit_config(config_path)

            with patch("diffaudit.attacks.pia_adapter.tv_datasets.CIFAR10", FakeCIFAR10):
                result = run_tmiadm_protocol_probe(
                    config=config,
                    workspace=root / "tmiadm-protocol-probe-jitter-defense",
                    repo_root=repo_root,
                    member_split_root=member_split_root,
                    device="cpu",
                    max_samples=2,
                    batch_size=2,
                    scan_timesteps=[10, 20, 30],
                    noise_seed=0,
                    timestep_jitter_radius=3,
                )

        self.assertTrue(result["defense"]["enabled"])
        self.assertEqual(result["defense"]["name"], "timestep-jitter")
        self.assertEqual(result["defense"]["timestep_jitter_radius"], 3)
        self.assertEqual(len(result["runtime"]["effective_timesteps"]), 3)

    def test_protocol_probe_accepts_temporal_striding_defense(self) -> None:
        from diffaudit.attacks.pia_adapter import bootstrap_pia_smoke_assets
        from diffaudit.attacks.tmiadm_adapter import run_tmiadm_protocol_probe
        from diffaudit.config import load_audit_config
        from tests.test_pia_adapter import PIA_CONFIG_TEMPLATE, create_minimal_pia_repo

        class FakeCIFAR10:
            def __init__(self, root, train, transform=None, download=False):
                del root, train, download
                self.transform = transform
                self.images = [
                    np.full((32, 32, 3), fill_value=value, dtype=np.uint8)
                    for value in (48, 80, 176, 208)
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
            config = load_audit_config(config_path)

            with patch("diffaudit.attacks.pia_adapter.tv_datasets.CIFAR10", FakeCIFAR10):
                result = run_tmiadm_protocol_probe(
                    config=config,
                    workspace=root / "tmiadm-protocol-probe-temporal-striding-defense",
                    repo_root=repo_root,
                    member_split_root=member_split_root,
                    device="cpu",
                    max_samples=2,
                    batch_size=2,
                    scan_timesteps=[10, 20, 30, 40],
                    noise_seed=0,
                    timestep_stride=2,
                )

        self.assertTrue(result["defense"]["enabled"])
        self.assertEqual(result["defense"]["name"], "temporal-striding")
        self.assertEqual(result["defense"]["timestep_stride"], 2)
        self.assertEqual(result["runtime"]["effective_timesteps"], [10, 30])


if __name__ == "__main__":
    unittest.main()
