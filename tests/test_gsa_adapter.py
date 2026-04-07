import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import torch


def create_minimal_gsa_repo(repo_root: Path) -> None:
    ddpm_root = repo_root / "DDPM"
    ddpm_root.mkdir(parents=True)
    (ddpm_root / "gen_l2_gradients_DDPM.py").write_text("# gradient entrypoint\n", encoding="utf-8")
    (ddpm_root / "train_unconditional.py").write_text("# train entrypoint\n", encoding="utf-8")
    (repo_root / "test_attack_accuracy.py").write_text("# attack entrypoint\n", encoding="utf-8")


def create_minimal_gsa_assets(assets_root: Path) -> None:
    for relative in (
        "datasets/target-member",
        "datasets/target-nonmember",
        "datasets/shadow-member",
        "datasets/shadow-nonmember",
        "checkpoints/target/checkpoint-10",
        "checkpoints/shadow/checkpoint-20",
        "manifests",
        "sources",
    ):
        (assets_root / relative).mkdir(parents=True, exist_ok=True)
    (assets_root / "datasets" / "target-member" / "sample-a.png").write_bytes(b"png")
    (assets_root / "datasets" / "target-nonmember" / "sample-b.png").write_bytes(b"png")
    (assets_root / "datasets" / "shadow-member" / "sample-c.png").write_bytes(b"png")
    (assets_root / "datasets" / "shadow-nonmember" / "sample-d.png").write_bytes(b"png")
    (assets_root / "manifests" / "split-manifest.json").write_text("{}", encoding="utf-8")
    (assets_root / "sources" / "cifar-10-python.tar.gz").write_bytes(b"archive")


class GsaAdapterTests(unittest.TestCase):
    def test_cli_probes_gsa_assets(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "GSA"
            assets_root = root / "assets"
            create_minimal_gsa_repo(repo_root)
            create_minimal_gsa_assets(assets_root)

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "probe-gsa-assets",
                        "--repo-root",
                        str(repo_root),
                        "--assets-root",
                        str(assets_root),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["checks"]["target_checkpoint"])
        self.assertTrue(payload["checks"]["shadow_checkpoint"])
        self.assertTrue(payload["checks"]["manifest_file"])

    def test_cli_runs_gsa_runtime_mainline(self) -> None:
        from diffaudit.cli import main

        def fake_subprocess_run(command, cwd=None, capture_output=None, text=None, check=None):
            del cwd, capture_output, text, check
            if "--output_name" in command:
                output_path = Path(command[command.index("--output_name") + 1])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                base = 0.9 if "member" in output_path.name else 0.1
                tensor = torch.tensor(
                    [
                        [base, base + 0.1, base + 0.2],
                        [base + 0.05, base + 0.15, base + 0.25],
                    ],
                    dtype=torch.float32,
                )
                torch.save(tensor, output_path)
                return type(
                    "Completed",
                    (),
                    {"returncode": 0, "stdout": "gradient ok\n", "stderr": ""},
                )()
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "attack ok\n", "stderr": ""},
            )()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "GSA"
            assets_root = root / "assets"
            create_minimal_gsa_repo(repo_root)
            create_minimal_gsa_assets(assets_root)

            stdout = StringIO()
            with patch("diffaudit.attacks.gsa.subprocess.run", side_effect=fake_subprocess_run):
                with redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "run-gsa-runtime-mainline",
                            "--workspace",
                            str(root / "gsa-runtime-mainline"),
                            "--repo-root",
                            str(repo_root),
                            "--assets-root",
                            str(assets_root),
                            "--resolution",
                            "32",
                            "--ddpm-num-steps",
                            "20",
                            "--sampling-frequency",
                            "2",
                            "--attack-method",
                            "1",
                        ]
                    )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["mode"], "runtime-mainline")
            self.assertEqual(payload["workspace_name"], "gsa-runtime-mainline")
            self.assertEqual(payload["evidence_level"], "runtime-mainline")
            self.assertEqual(payload["asset_grade"], "real-asset-closed-loop")
            self.assertEqual(payload["contract_stage"], "target")
            self.assertIn("auc", payload["metrics"])
            self.assertIn("asr", payload["metrics"])
            self.assertIn("tpr_at_1pct_fpr", payload["metrics"])
            self.assertIn("tpr_at_0_1pct_fpr", payload["metrics"])
            self.assertTrue(Path(payload["artifact_paths"]["summary"]).exists())
            self.assertTrue(Path(payload["artifact_paths"]["target_member_gradients"]).exists())


if __name__ == "__main__":
    unittest.main()
