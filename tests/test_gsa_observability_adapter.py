import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


def create_minimal_gsa_repo(repo_root: Path) -> None:
    ddpm_root = repo_root / "DDPM"
    ddpm_root.mkdir(parents=True)
    (ddpm_root / "gen_l2_gradients_DDPM.py").write_text("# gradient entrypoint\n", encoding="utf-8")
    (ddpm_root / "train_unconditional.py").write_text("# train entrypoint\n", encoding="utf-8")
    (repo_root / "test_attack_accuracy.py").write_text("# attack entrypoint\n", encoding="utf-8")


def create_minimal_observability_assets(assets_root: Path) -> None:
    for relative in (
        "datasets/target-member",
        "datasets/target-nonmember",
        "checkpoints/target/checkpoint-9600",
    ):
        (assets_root / relative).mkdir(parents=True, exist_ok=True)
    (assets_root / "datasets" / "target-member" / "00-data_batch_1-00965.png").write_bytes(b"png")
    (assets_root / "datasets" / "target-nonmember" / "00-data_batch_1-00467.png").write_bytes(b"png")


class GsaObservabilityAdapterTests(unittest.TestCase):
    def test_probe_resolves_canonical_sample_binding(self) -> None:
        from diffaudit.attacks.gsa_observability import probe_gsa_observability_contract

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "GSA"
            assets_root = root / "assets"
            create_minimal_gsa_repo(repo_root)
            create_minimal_observability_assets(assets_root)

            payload = probe_gsa_observability_contract(
                repo_root=repo_root,
                assets_root=assets_root,
                checkpoint_root=assets_root / "checkpoints" / "target",
                split="target-member",
                sample_id="target-member/00-data_batch_1-00965.png",
                layer_selector="mid_block.attentions.0.to_v",
                signal_type="activations",
            )

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(
            payload["resolved"]["sample_binding"]["dataset_relpath"],
            "00-data_batch_1-00965.png",
        )
        self.assertEqual(
            payload["resolved"]["layer_binding"]["layer_id"],
            "mid_block.attentions.0.to_v",
        )
        self.assertTrue(payload["checks"]["resolved_checkpoint_exists"])

    def test_probe_accepts_legacy_sample_alias(self) -> None:
        from diffaudit.attacks.gsa_observability import resolve_gsa_sample_binding

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_root = Path(tmpdir) / "assets"
            create_minimal_observability_assets(assets_root)
            binding = resolve_gsa_sample_binding(
                assets_root=assets_root,
                split="target-member",
                sample_id="target-member:00-data_batch_1-00965",
            )

        self.assertEqual(binding["sample_id"], "target-member/00-data_batch_1-00965.png")
        self.assertEqual(binding["binding_source"], "filesystem-scan")

    def test_cli_probes_gsa_observability_contract(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "GSA"
            assets_root = root / "assets"
            create_minimal_gsa_repo(repo_root)
            create_minimal_observability_assets(assets_root)

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "probe-gsa-observability-contract",
                        "--repo-root",
                        str(repo_root),
                        "--assets-root",
                        str(assets_root),
                        "--checkpoint-root",
                        str(assets_root / "checkpoints" / "target"),
                        "--split",
                        "target-member",
                        "--sample-id",
                        "target-member/00-data_batch_1-00965.png",
                        "--layer-selector",
                        "mid_block.attentions.0.to_v",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["mode"], "contract-probe")
        self.assertEqual(payload["gpu_release"], "none")


if __name__ == "__main__":
    unittest.main()
