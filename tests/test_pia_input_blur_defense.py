import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image


class PiaInputBlurDefenseTests(unittest.TestCase):
    def test_apply_input_defense_can_blur_batch(self) -> None:
        from diffaudit.attacks.pia_adapter import _apply_input_defense

        batch = torch.zeros(1, 3, 8, 8)
        batch[:, :, 4, 4] = 1.0

        blurred = _apply_input_defense(batch, input_gaussian_blur_sigma=1.0)

        self.assertFalse(torch.equal(blurred, batch))

    def test_runtime_mainline_accepts_input_blur_defense(self) -> None:
        from diffaudit.attacks.pia_adapter import bootstrap_pia_smoke_assets, run_pia_runtime_mainline
        from diffaudit.config import load_audit_config
        from tests.test_pia_adapter import create_minimal_pia_repo

        class FakeCIFAR10:
            def __init__(self, root, train, transform=None, download=False):
                del root, train, download
                self.transform = transform
                self.images = [
                    np.full((32, 32, 3), fill_value=value, dtype=np.uint8)
                    for value in (32, 64, 96, 128)
                ]

            def __len__(self) -> int:
                return len(self.images)

            def __getitem__(self, index: int):
                image = Image.fromarray(self.images[index])
                tensor = self.transform(image) if self.transform is not None else image
                return tensor, index

        config_template = """
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
  num_samples: 2
  parameters:
    attacker_name: PIA
    attack_num: 3
    interval: 10
report:
  output_dir: experiments/pia-runtime
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
                result = run_pia_runtime_mainline(
                    config=config,
                    workspace=root / "pia-runtime-blur-defense",
                    repo_root=repo_root,
                    member_split_root=member_split_root,
                    device="cpu",
                    max_samples=2,
                    batch_size=2,
                    input_gaussian_blur_sigma=1.0,
                )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["defense"]["name"], "input-gaussian-blur")


if __name__ == "__main__":
    unittest.main()
