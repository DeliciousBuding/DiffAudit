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


class FakeAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_v = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.proj_v.weight.zero_()
            self.proj.weight.zero_()
            for channel in range(3):
                self.proj_v.weight[channel, channel, 0, 0] = 1.0
                self.proj.weight[channel, channel, 0, 0] = 1.0

    def forward(self, x):
        return self.proj(torch.relu(self.proj_v(x)))


class FakeMiddleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = FakeAttn()

    def forward(self, x, temb=None):
        del temb
        return x + self.attn(x)


class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        del T, ch, ch_mult, attn, num_res_blocks
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout2d(p=dropout)
        self.middleblocks = nn.ModuleList([FakeMiddleBlock()])

    def forward(self, x, t):
        del t
        h = self.dropout(x) * self.scale
        for block in self.middleblocks:
            h = block(h)
        return h
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

    def test_cli_runs_pia_runtime_mainline(self) -> None:
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
                    for value in (16, 24, 32, 40, 180, 188, 196, 204)
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
                mia_train_idxs=np.array([0, 1, 2, 3]),
                mia_eval_idxs=np.array([4, 5, 6, 7]),
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
                            "run-pia-runtime-mainline",
                            "--config",
                            str(config_path),
                            "--workspace",
                            str(root / "pia-runtime-mainline"),
                            "--repo-root",
                            str(repo_root),
                            "--member-split-root",
                            str(member_split_root),
                            "--device",
                            "cpu",
                            "--max-samples",
                            "4",
                            "--batch-size",
                            "2",
                            "--stochastic-dropout-defense",
                            "--dropout-activation-schedule",
                            "late_steps_only",
                            "--adaptive-query-repeats",
                            "3",
                            "--provenance-status",
                            "workspace-verified",
                        ]
                    )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["mode"], "runtime-mainline")
            self.assertEqual(payload["workspace_name"], "pia-runtime-mainline")
            self.assertEqual(payload["evidence_level"], "runtime-mainline")
            self.assertEqual(payload["asset_grade"], "single-machine-real-asset")
            self.assertEqual(payload["provenance_status"], "workspace-verified")
            self.assertIn("auc", payload["metrics"])
            self.assertIn("asr", payload["metrics"])
            self.assertIn("tpr_at_1pct_fpr", payload["metrics"])
            self.assertIn("tpr_at_0_1pct_fpr", payload["metrics"])
            self.assertEqual(payload["defense"]["name"], "stochastic-dropout")
            self.assertEqual(payload["defense"]["dropout_activation_schedule"], "late_steps_only")
            self.assertTrue(payload["defense"]["enabled"])
            self.assertEqual(payload["defense_stage"], "provisional-g1")
            self.assertEqual(payload["adaptive_check"]["query_repeats"], 3)
            self.assertEqual(payload["adaptive_check"]["aggregation"], "mean")
            self.assertIn("auc", payload["adaptive_check"]["metrics"])
            self.assertIn("asr", payload["adaptive_check"]["metrics"])
            self.assertIn("fid", payload["quality"]["metrics"])
            self.assertIn("lpips", payload["quality"]["metrics"])
            self.assertEqual(payload["quality"]["suite"], "pia-runtime-surrogate-v1")
            self.assertEqual(payload["cost"]["adaptive_query_repeats"], 3)
            self.assertEqual(payload["cost"]["queries_per_sample"], 3)
            self.assertEqual(payload["runtime"]["max_samples"], 4)
            self.assertEqual(payload["runtime"]["num_samples"], 4)
            self.assertEqual(payload["sample_count_per_split"], 4)
            self.assertTrue(Path(payload["artifact_paths"]["summary"]).exists())

    def test_cli_runs_pia_runtime_mainline_with_precision_throttling_defense(self) -> None:
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
                    for value in (16, 24, 32, 40, 180, 188, 196, 204)
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
                mia_train_idxs=np.array([0, 1, 2, 3]),
                mia_eval_idxs=np.array([4, 5, 6, 7]),
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
                            "run-pia-runtime-mainline",
                            "--config",
                            str(config_path),
                            "--workspace",
                            str(root / "pia-runtime-mainline-precision"),
                            "--repo-root",
                            str(repo_root),
                            "--member-split-root",
                            str(member_split_root),
                            "--device",
                            "cpu",
                            "--max-samples",
                            "4",
                            "--batch-size",
                            "2",
                            "--epsilon-precision-bins",
                            "32",
                            "--provenance-status",
                            "workspace-verified",
                        ]
                    )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["defense"]["name"], "epsilon-precision-throttling")
            self.assertEqual(payload["defense"]["epsilon_precision_bins"], 32)
            self.assertFalse(payload["defense"]["enabled"] is False)
            self.assertEqual(payload["defense_stage"], "candidate-g2")

    def test_cli_exports_pia_translated_alias_probe(self) -> None:
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
                    for value in (24, 208)
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
                mia_train_idxs=np.array([0]),
                mia_eval_idxs=np.array([1]),
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
                            "export-pia-translated-alias-probe",
                            "--config",
                            str(config_path),
                            "--workspace",
                            str(root / "pia-translated-alias-probe"),
                            "--repo-root",
                            str(repo_root),
                            "--member-split-root",
                            str(member_split_root),
                            "--device",
                            "cpu",
                            "--member-index",
                            "0",
                            "--nonmember-index",
                            "1",
                            "--batch-size",
                            "1",
                            "--adaptive-query-repeats",
                            "2",
                        ]
                    )

            payload = json.loads(stdout.getvalue())
            sample_scores_path = Path(payload["artifact_paths"]["sample_scores"])
            self.assertTrue(sample_scores_path.exists())
            records = [
                json.loads(line)
                for line in sample_scores_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(records), 2)
            self.assertTrue(any(abs(float(record["score_delta"])) > 0.0 for record in records))

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["mode"], "translated-alias-probe")
        self.assertEqual(payload["translation"]["alias_selector"], "middleblocks.0.attn.proj_v")
        self.assertEqual(payload["translation"]["translated_from"], "mid_block.attentions.0.to_v")
        self.assertEqual(payload["translation"]["translation_kind"], "translated-contract")
        self.assertTrue(payload["translation"]["translation_not_same_spec"])
        self.assertEqual(payload["translation"]["channel_dim"], 1)
        self.assertEqual(payload["translation"]["tensor_layout"], "BCHW")
        self.assertGreater(payload["translation"]["hook_hits"]["baseline_score_path"], 0)
        self.assertGreater(payload["translation"]["hook_hits"]["intervened_score_path"], 0)
        self.assertEqual(payload["packet"]["member_indices"], [0])
        self.assertEqual(payload["packet"]["nonmember_indices"], [1])
        self.assertIn("sample_scores", payload["artifact_paths"])


if __name__ == "__main__":
    unittest.main()
