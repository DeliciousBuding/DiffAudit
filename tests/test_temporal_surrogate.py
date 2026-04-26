import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


class TemporalSurrogateTests(unittest.TestCase):
    def test_compute_temporal_feature_bank_returns_named_features(self) -> None:
        import torch

        from diffaudit.attacks.temporal_surrogate import compute_temporal_feature_bank

        timestep_predictions = torch.tensor(
            [
                [
                    [[[0.1, 0.2], [0.2, 0.1]]],
                    [[[0.2, 0.3], [0.3, 0.2]]],
                    [[[0.4, 0.5], [0.5, 0.4]]],
                    [[[0.7, 0.8], [0.8, 0.7]]],
                    [[[0.9, 1.0], [1.0, 0.9]]],
                    [[[1.1, 1.2], [1.2, 1.1]]],
                ],
                [
                    [[[0.3, 0.3], [0.3, 0.3]]],
                    [[[0.32, 0.32], [0.32, 0.32]]],
                    [[[0.34, 0.34], [0.34, 0.34]]],
                    [[[0.36, 0.36], [0.36, 0.36]]],
                    [[[0.38, 0.38], [0.38, 0.38]]],
                    [[[0.4, 0.4], [0.4, 0.4]]],
                ],
            ],
            dtype=torch.float32,
        )

        feature_names, features = compute_temporal_feature_bank(
            timestep_predictions=timestep_predictions,
            timesteps=[10, 20, 30, 40, 50, 60],
        )

        self.assertEqual(features.shape[0], 2)
        self.assertEqual(features.shape[1], len(feature_names))
        self.assertIn("eps_abs_mean_early", feature_names)
        self.assertIn("eps_abs_mean_late", feature_names)
        self.assertIn("eps_abs_late_over_early", feature_names)
        self.assertIn("eps_abs_curvature", feature_names)
        late_mean_index = feature_names.index("eps_abs_mean_late")
        self.assertGreater(features[0, late_mean_index], features[1, late_mean_index])

    def test_export_temporal_surrogate_feature_packet_runs_on_minimal_pia_assets(self) -> None:
        from diffaudit.attacks.pia_adapter import bootstrap_pia_smoke_assets
        from diffaudit.attacks.temporal_surrogate import export_temporal_surrogate_feature_packet
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
                result = export_temporal_surrogate_feature_packet(
                    config=config,
                    workspace=root / "temporal-surrogate-feature-packet",
                    repo_root=repo_root,
                    member_split_root=member_split_root,
                    device="cpu",
                    max_samples=2,
                    batch_size=2,
                    scan_timesteps=[10, 20, 30, 40],
                    noise_seed=0,
                )

            feature_packet = json.loads(
                (root / "temporal-surrogate-feature-packet" / "feature-packet.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["method"], "temporal-surrogate")
        self.assertEqual(result["mode"], "feature-packet-export")
        self.assertEqual(feature_packet["member_indices"], [0, 1])
        self.assertEqual(feature_packet["nonmember_indices"], [2, 3])
        self.assertEqual(len(feature_packet["feature_names"]), len(feature_packet["member_features"][0]))

    def test_evaluate_temporal_surrogate_packets_reports_teacher_and_transfer_metrics(self) -> None:
        from diffaudit.attacks.temporal_surrogate import evaluate_temporal_surrogate_packets

        rng = np.random.default_rng(7)
        feature_names = [
            "eps_abs_mean_early",
            "eps_abs_mean_mid",
            "eps_abs_mean_late",
            "eps_abs_late_minus_early",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            teacher_packet = root / "teacher-feature-packet.json"
            transfer_packet = root / "transfer-feature-packet.json"
            teacher_surface = root / "teacher-surface.json"

            teacher_member_features = rng.normal(loc=1.0, scale=0.2, size=(24, 4))
            teacher_nonmember_features = rng.normal(loc=-1.0, scale=0.2, size=(24, 4))
            transfer_member_features = rng.normal(loc=1.1, scale=0.25, size=(20, 4))
            transfer_nonmember_features = rng.normal(loc=-0.9, scale=0.25, size=(20, 4))

            def score_from_features(features: np.ndarray) -> np.ndarray:
                return 0.7 * features[:, 0] + 0.5 * features[:, 2] + 0.2 * features[:, 3]

            teacher_member_scores = score_from_features(teacher_member_features) + rng.normal(0.0, 0.03, size=24)
            teacher_nonmember_scores = score_from_features(teacher_nonmember_features) + rng.normal(0.0, 0.03, size=24)

            _write_json(
                teacher_packet,
                {
                    "feature_names": feature_names,
                    "member_indices": list(range(100, 124)),
                    "nonmember_indices": list(range(200, 224)),
                    "member_features": np.round(teacher_member_features, 6).tolist(),
                    "nonmember_features": np.round(teacher_nonmember_features, 6).tolist(),
                },
            )
            _write_json(
                transfer_packet,
                {
                    "feature_names": feature_names,
                    "member_indices": list(range(300, 320)),
                    "nonmember_indices": list(range(400, 420)),
                    "member_features": np.round(transfer_member_features, 6).tolist(),
                    "nonmember_features": np.round(transfer_nonmember_features, 6).tolist(),
                },
            )
            _write_json(
                teacher_surface,
                {
                    "member_scores": np.round(teacher_member_scores[::-1], 6).tolist(),
                    "nonmember_scores": np.round(teacher_nonmember_scores[::-1], 6).tolist(),
                    "member_indices": list(range(123, 99, -1)),
                    "nonmember_indices": list(range(223, 199, -1)),
                },
            )

            result = evaluate_temporal_surrogate_packets(
                workspace=root / "temporal-surrogate-eval",
                teacher_feature_packet=teacher_packet,
                teacher_score_surface=teacher_surface,
                transfer_feature_packet=transfer_packet,
                bag_count=6,
                quantiles=[0.2, 0.4, 0.6, 0.8],
                cv_splits=4,
                cv_repeats=2,
                random_seed=5,
            )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["mode"], "teacher-calibrated-temporal-surrogate")
        self.assertGreater(result["teacher"]["metrics"]["spearman"], 0.8)
        self.assertGreater(result["teacher"]["metrics"]["pearson"], 0.8)
        self.assertIn("threshold_cv", result["teacher"]["metrics"])
        self.assertTrue(result["teacher"]["hard_gates"]["spearman_ge_0_8"])
        self.assertIn("transfer", result)
        self.assertGreater(result["transfer"]["metrics"]["auc"], 0.9)
        self.assertIn("shift_audit", result["transfer"])


if __name__ == "__main__":
    unittest.main()
