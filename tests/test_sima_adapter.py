import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image


class SimaAdapterTests(unittest.TestCase):
    def test_compute_sima_scores_prefers_lower_norm_predictions(self) -> None:
        from diffaudit.attacks.sima_adapter import _compute_sima_scores_from_eps

        eps_prediction = torch.tensor(
            [
                [[[1.0, 0.0], [0.0, 0.0]]],
                [[[2.0, 0.0], [0.0, 0.0]]],
            ]
        )

        scores = _compute_sima_scores_from_eps(eps_prediction, p_norm=2)

        self.assertEqual(list(scores.shape), [2])
        self.assertGreater(scores[0].item(), scores[1].item())

    def test_runtime_feasibility_runs_on_minimal_pia_assets(self) -> None:
        from diffaudit.attacks.sima_adapter import run_sima_runtime_feasibility
        from diffaudit.attacks.pia_adapter import bootstrap_pia_smoke_assets
        from diffaudit.config import load_audit_config
        from tests.test_pia_adapter import create_minimal_pia_repo

        class FakeCIFAR10:
            def __init__(self, root, train, transform=None, download=False):
                del root, train, download
                self.transform = transform
                self.images = [
                    np.full((32, 32, 3), fill_value=value, dtype=np.uint8)
                    for value in (16, 48, 144, 208)
                ]

            def __len__(self) -> int:
                return len(self.images)

            def __getitem__(self, index: int):
                image = Image.fromarray(self.images[index])
                tensor = self.transform(image) if self.transform is not None else image
                return tensor, index

        config_template = """
task:
  name: sima-runtime
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
  num_samples: 2
  parameters:
    attacker_name: PIA
    attack_num: 3
    interval: 10
report:
  output_dir: experiments/sima-runtime
"""

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
                config_template
                .replace("PLACEHOLDER_DATASET", str(dataset_root).replace("\\", "/"))
                .replace("PLACEHOLDER_MODEL", str(model_dir).replace("\\", "/")),
                encoding="utf-8",
            )
            config = load_audit_config(config_path)

            with patch("diffaudit.attacks.pia_adapter.tv_datasets.CIFAR10", FakeCIFAR10):
                result = run_sima_runtime_feasibility(
                    config=config,
                    workspace=root / "sima-runtime-feasibility",
                    repo_root=repo_root,
                    member_split_root=member_split_root,
                    device="cpu",
                    max_samples=2,
                    batch_size=2,
                    scan_timesteps=[10, 20],
                    p_norm=2,
                    noise_seed=0,
                )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["method"], "sima")
        self.assertEqual(result["best_timestep"], 10)

    def test_export_sima_packet_scores_emits_pairboard_ready_surface(self) -> None:
        from diffaudit.attacks.pia_adapter import bootstrap_pia_smoke_assets
        from diffaudit.attacks.sima_adapter import export_sima_packet_scores
        from diffaudit.config import load_audit_config
        from tests.test_pia_adapter import create_minimal_pia_repo

        class FakeCIFAR10:
            def __init__(self, root, train, transform=None, download=False):
                del root, train, download
                self.transform = transform
                self.images = [
                    np.full((32, 32, 3), fill_value=value, dtype=np.uint8)
                    for value in (12, 36, 60, 168, 192, 216)
                ]

            def __len__(self) -> int:
                return len(self.images)

            def __getitem__(self, index: int):
                image = Image.fromarray(self.images[index])
                tensor = self.transform(image) if self.transform is not None else image
                return tensor, index

        config_template = """
task:
  name: sima-packet
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
  num_samples: 3
  parameters:
    attacker_name: PIA
    attack_num: 3
    interval: 10
report:
  output_dir: experiments/sima-packet
"""

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
                mia_train_idxs=np.array([0, 1, 2]),
                mia_eval_idxs=np.array([3, 4, 5]),
            )

            model_dir = root / "model"
            bootstrap_pia_smoke_assets(target_dir=model_dir, repo_root=repo_root)

            config_path = root / "audit.yaml"
            config_path.write_text(
                config_template
                .replace("PLACEHOLDER_DATASET", str(dataset_root).replace("\\", "/"))
                .replace("PLACEHOLDER_MODEL", str(model_dir).replace("\\", "/")),
                encoding="utf-8",
            )
            config = load_audit_config(config_path)

            member_index_file = root / "member-indices.json"
            nonmember_index_file = root / "nonmember-indices.txt"
            member_index_file.write_text(json.dumps([2, 0]), encoding="utf-8")
            nonmember_index_file.write_text("5\n3\n4\n", encoding="utf-8")

            with patch("diffaudit.attacks.pia_adapter.tv_datasets.CIFAR10", FakeCIFAR10):
                payload = export_sima_packet_scores(
                    config=config,
                    workspace=root / "sima-packet-export",
                    repo_root=repo_root,
                    member_split_root=member_split_root,
                    device="cpu",
                    packet_size=1,
                    member_offset=1,
                    nonmember_offset=1,
                    member_index_file=member_index_file,
                    nonmember_index_file=nonmember_index_file,
                    batch_size=2,
                    timestep=20,
                    p_norm=2,
                    noise_seed=0,
                )

            scores_path = Path(payload["artifact_paths"]["scores"])
            sample_scores_path = Path(payload["artifact_paths"]["sample_scores"])
            self.assertTrue(scores_path.exists())
            self.assertTrue(sample_scores_path.exists())
            scores_payload = json.loads(scores_path.read_text(encoding="utf-8"))
            records = [
                json.loads(line)
                for line in sample_scores_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["track"], "gray-box")
        self.assertEqual(payload["method"], "sima")
        self.assertEqual(payload["mode"], "packet-score-export")
        self.assertEqual(payload["device"], "cpu")
        self.assertEqual(payload["gpu_release"], "none")
        self.assertEqual(payload["admitted_change"], "none")
        self.assertEqual(payload["runtime"]["selection_mode"], "explicit-index-files")
        self.assertEqual(payload["runtime"]["timestep"], 20)
        self.assertEqual(payload["runtime"]["member_packet_size"], 2)
        self.assertEqual(payload["runtime"]["nonmember_packet_size"], 3)
        self.assertEqual(payload["packet"]["member_indices"], [2, 0])
        self.assertEqual(payload["packet"]["nonmember_indices"], [5, 3, 4])
        self.assertEqual(payload["checks"]["sample_scores_written"], 5)
        self.assertTrue(payload["checks"]["packet_scores_generated"])
        self.assertTrue(payload["checks"]["device_cpu_only"])
        self.assertEqual(scores_payload["member_indices"], [2, 0])
        self.assertEqual(scores_payload["nonmember_indices"], [5, 3, 4])
        self.assertEqual(len(scores_payload["member_scores"]), 2)
        self.assertEqual(len(scores_payload["nonmember_scores"]), 3)
        self.assertEqual(len(records), 5)
        self.assertEqual(records[0]["membership"], "member")
        self.assertEqual(records[-1]["membership"], "nonmember")

    def test_cli_exports_sima_packet_scores(self) -> None:
        from diffaudit.attacks.pia_adapter import bootstrap_pia_smoke_assets
        from diffaudit.cli import main
        from tests.test_pia_adapter import create_minimal_pia_repo

        class FakeCIFAR10:
            def __init__(self, root, train, transform=None, download=False):
                del root, train, download
                self.transform = transform
                self.images = [
                    np.full((32, 32, 3), fill_value=value, dtype=np.uint8)
                    for value in (18, 42, 66, 150, 174, 198)
                ]

            def __len__(self) -> int:
                return len(self.images)

            def __getitem__(self, index: int):
                image = Image.fromarray(self.images[index])
                tensor = self.transform(image) if self.transform is not None else image
                return tensor, index

        config_template = """
task:
  name: sima-packet-cli
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
  num_samples: 3
  parameters:
    attacker_name: PIA
    attack_num: 3
    interval: 10
report:
  output_dir: experiments/sima-packet-cli
"""

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
                mia_train_idxs=np.array([0, 1, 2]),
                mia_eval_idxs=np.array([3, 4, 5]),
            )

            model_dir = root / "model"
            bootstrap_pia_smoke_assets(target_dir=model_dir, repo_root=repo_root)

            config_path = root / "audit.yaml"
            config_path.write_text(
                config_template
                .replace("PLACEHOLDER_DATASET", str(dataset_root).replace("\\", "/"))
                .replace("PLACEHOLDER_MODEL", str(model_dir).replace("\\", "/")),
                encoding="utf-8",
            )

            stdout = StringIO()
            with patch("diffaudit.attacks.pia_adapter.tv_datasets.CIFAR10", FakeCIFAR10):
                with redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "export-sima-packet-scores",
                            "--config",
                            str(config_path),
                            "--workspace",
                            str(root / "sima-packet-cli-export"),
                            "--repo-root",
                            str(repo_root),
                            "--member-split-root",
                            str(member_split_root),
                            "--device",
                            "cpu",
                            "--packet-size",
                            "2",
                            "--member-offset",
                            "1",
                            "--nonmember-offset",
                            "0",
                            "--batch-size",
                            "2",
                            "--timestep",
                            "20",
                            "--p-norm",
                            "2",
                            "--noise-seed",
                            "0",
                        ]
                    )

            payload = json.loads(stdout.getvalue())
            scores_path = Path(payload["artifact_paths"]["scores"])
            self.assertTrue(scores_path.exists())
            scores_payload = json.loads(scores_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["method"], "sima")
        self.assertEqual(payload["mode"], "packet-score-export")
        self.assertEqual(payload["runtime"]["selection_mode"], "offset-slice")
        self.assertEqual(payload["packet"]["member_indices"], [1, 2])
        self.assertEqual(payload["packet"]["nonmember_indices"], [3, 4])
        self.assertEqual(len(scores_payload["member_scores"]), 2)
        self.assertEqual(len(scores_payload["nonmember_scores"]), 2)


if __name__ == "__main__":
    unittest.main()
