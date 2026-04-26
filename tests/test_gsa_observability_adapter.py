import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file


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


def create_export_ready_observability_assets(assets_root: Path) -> None:
    for relative in (
        "datasets/target-member",
        "datasets/target-nonmember",
        "checkpoints/target/checkpoint-9600",
    ):
        (assets_root / relative).mkdir(parents=True, exist_ok=True)

    for relative, color in (
        ("datasets/target-member/00-data_batch_1-00965.png", (255, 0, 0)),
        ("datasets/target-nonmember/00-data_batch_1-00467.png", (0, 255, 0)),
        ("datasets/target-nonmember/00-data_batch_1-01278.png", (0, 0, 255)),
    ):
        Image.new("RGB", (32, 32), color=color).save(assets_root / relative)

    from diffaudit.attacks.gsa_observability import _build_gsa_unet

    model = _build_gsa_unet()
    state_dict = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    save_file(
        state_dict,
        str(assets_root / "checkpoints" / "target" / "checkpoint-9600" / "model.safetensors"),
    )


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

    def test_export_writes_activation_artifacts_for_sample_pair(self) -> None:
        from diffaudit.attacks.gsa_observability import export_gsa_observability_canary

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "GSA"
            assets_root = root / "assets"
            workspace = root / "workspace"
            create_minimal_gsa_repo(repo_root)
            create_export_ready_observability_assets(assets_root)

            payload = export_gsa_observability_canary(
                workspace=workspace,
                repo_root=repo_root,
                assets_root=assets_root,
                checkpoint_root=assets_root / "checkpoints" / "target",
                checkpoint_dir=assets_root / "checkpoints" / "target" / "checkpoint-9600",
                split="target-member",
                sample_id="target-member/00-data_batch_1-00965.png",
                control_split="target-nonmember",
                control_sample_id="target-nonmember/00-data_batch_1-00467.png",
                layer_selector="mid_block.attentions.0.to_v",
                signal_type="activations",
                timestep=999,
                noise_seed=7,
                prediction_type="epsilon",
                device="cpu",
            )

            summary_path = workspace / "summary.json"
            records_path = workspace / "records.jsonl"

            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["mode"], "activation-export-canary")
            self.assertEqual(payload["gpu_release"], "none")
            self.assertTrue(summary_path.exists())
            self.assertTrue(records_path.exists())

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["checks"]["records_written"], 2)
            self.assertEqual(summary["checks"]["tensor_artifacts_written"], 2)
            self.assertFalse(Path(summary["artifact_paths"]["records"]).is_absolute())

            records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 2)
            for record in records:
                self.assertIn(record["split"], {"target-member", "target-nonmember"})
                self.assertEqual(record["signal_type"], "activations")
                self.assertEqual(record["layer_id"], "mid_block.attentions.0.to_v")
                self.assertEqual(record["timestep"], 999)
                self.assertFalse(Path(record["artifact_path"]).is_absolute())
                artifact_path = workspace / record["artifact_path"]
                self.assertTrue(artifact_path.exists())
                tensor = torch.load(artifact_path, map_location="cpu", weights_only=False)
                self.assertEqual(tuple(tensor.shape), tuple(record["tensor_shape"]))

    def test_cli_exports_activation_canary_with_cpu_only_contract(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "GSA"
            assets_root = root / "assets"
            workspace = root / "workspace"
            create_minimal_gsa_repo(repo_root)
            create_export_ready_observability_assets(assets_root)

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "export-gsa-observability-canary",
                        "--workspace",
                        str(workspace),
                        "--repo-root",
                        str(repo_root),
                        "--assets-root",
                        str(assets_root),
                        "--checkpoint-root",
                        str(assets_root / "checkpoints" / "target"),
                        "--checkpoint-dir",
                        str(assets_root / "checkpoints" / "target" / "checkpoint-9600"),
                        "--split",
                        "target-member",
                        "--sample-id",
                        "target-member/00-data_batch_1-00965.png",
                        "--control-split",
                        "target-nonmember",
                        "--control-sample-id",
                        "target-nonmember/00-data_batch_1-00467.png",
                        "--layer-selector",
                        "mid_block.attentions.0.to_v",
                        "--timestep",
                        "999",
                        "--noise-seed",
                        "7",
                        "--prediction-type",
                        "epsilon",
                        "--device",
                        "cpu",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["mode"], "activation-export-canary")
        self.assertEqual(payload["gpu_release"], "none")
        self.assertEqual(payload["requested"]["device"], "cpu")

    def test_export_writes_masked_packet_artifacts(self) -> None:
        from diffaudit.attacks.gsa_observability import export_gsa_observability_masked_packet

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "GSA"
            assets_root = root / "assets"
            workspace = root / "workspace"
            create_minimal_gsa_repo(repo_root)
            create_export_ready_observability_assets(assets_root)

            payload = export_gsa_observability_masked_packet(
                workspace=workspace,
                repo_root=repo_root,
                assets_root=assets_root,
                checkpoint_root=assets_root / "checkpoints" / "target",
                checkpoint_dir=assets_root / "checkpoints" / "target" / "checkpoint-9600",
                split="target-member",
                sample_id="target-member/00-data_batch_1-00965.png",
                control_split="target-nonmember",
                control_sample_id="target-nonmember/00-data_batch_1-00467.png",
                layer_selector="mid_block.attentions.0.to_v",
                mask_kind="top_abs_delta_k",
                k=4,
                alpha=0.5,
                timestep=999,
                noise_seed=7,
                mask_seed=11,
                device="cpu",
            )

            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["mode"], "masked-packet-export")
            self.assertEqual(payload["gpu_release"], "none")
            self.assertEqual(payload["mask"]["k"], 4)
            self.assertIn("selected_delta_retention_ratio", payload["metrics"])
            records = [json.loads(line) for line in (workspace / "records.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 2)
            for record in records:
                self.assertIn("pre_mask", record["artifact_paths"])
                self.assertIn("post_mask", record["artifact_paths"])
                self.assertTrue((workspace / record["artifact_paths"]["pre_mask"]).exists())
                self.assertTrue((workspace / record["artifact_paths"]["post_mask"]).exists())

    def test_export_writes_inmodel_packet_artifacts(self) -> None:
        from diffaudit.attacks.gsa_observability import export_gsa_observability_inmodel_packet

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "GSA"
            assets_root = root / "assets"
            workspace = root / "workspace"
            create_minimal_gsa_repo(repo_root)
            create_export_ready_observability_assets(assets_root)

            payload = export_gsa_observability_inmodel_packet(
                workspace=workspace,
                repo_root=repo_root,
                assets_root=assets_root,
                checkpoint_root=assets_root / "checkpoints" / "target",
                checkpoint_dir=assets_root / "checkpoints" / "target" / "checkpoint-9600",
                split="target-member",
                sample_id="target-member/00-data_batch_1-00965.png",
                control_split="target-nonmember",
                control_sample_id="target-nonmember/00-data_batch_1-01278.png",
                layer_selector="mid_block.attentions.0.to_v",
                mask_kind="top_abs_delta_k",
                k=4,
                alpha=0.0,
                timestep=999,
                noise_seed=7,
                mask_seed=11,
                device="cpu",
            )

            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["mode"], "inmodel-packet-export")
            self.assertEqual(payload["gpu_release"], "none")
            self.assertTrue(payload["checks"]["intervention_applied"])
            self.assertGreater(payload["metrics"]["epsilon_prediction_rms_drift_mean"], 0.0)
            records = [json.loads(line) for line in (workspace / "records.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 2)
            for record in records:
                self.assertIn("baseline_layer", record["artifact_paths"])
                self.assertIn("intervened_layer", record["artifact_paths"])
                self.assertIn("baseline_prediction", record["artifact_paths"])
                self.assertIn("intervened_prediction", record["artifact_paths"])
                self.assertTrue((workspace / record["artifact_paths"]["baseline_layer"]).exists())
                self.assertTrue((workspace / record["artifact_paths"]["intervened_layer"]).exists())
                self.assertTrue((workspace / record["artifact_paths"]["baseline_prediction"]).exists())
                self.assertTrue((workspace / record["artifact_paths"]["intervened_prediction"]).exists())

    def test_cli_exports_inmodel_packet_with_cpu_only_contract(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "GSA"
            assets_root = root / "assets"
            workspace = root / "workspace"
            create_minimal_gsa_repo(repo_root)
            create_export_ready_observability_assets(assets_root)

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "export-gsa-observability-inmodel-packet",
                        "--workspace",
                        str(workspace),
                        "--repo-root",
                        str(repo_root),
                        "--assets-root",
                        str(assets_root),
                        "--checkpoint-root",
                        str(assets_root / "checkpoints" / "target"),
                        "--checkpoint-dir",
                        str(assets_root / "checkpoints" / "target" / "checkpoint-9600"),
                        "--split",
                        "target-member",
                        "--sample-id",
                        "target-member/00-data_batch_1-00965.png",
                        "--control-split",
                        "target-nonmember",
                        "--control-sample-id",
                        "target-nonmember/00-data_batch_1-01278.png",
                        "--layer-selector",
                        "mid_block.attentions.0.to_v",
                        "--timestep",
                        "999",
                        "--noise-seed",
                        "7",
                        "--mask-seed",
                        "11",
                        "--prediction-type",
                        "epsilon",
                        "--device",
                        "cpu",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["mode"], "inmodel-packet-export")
        self.assertEqual(payload["requested"]["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
