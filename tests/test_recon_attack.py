import json
import importlib.util
import sys
import tempfile
import types
import unittest
import uuid
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

_KANDINSKY_INFERENCE_MODULE_CACHE = None


RECON_CONFIG_TEMPLATE = """
task:
  name: recon-plan
  model_family: diffusion
  access_level: black_box
assets:
  dataset_id: toy-dataset
  dataset_root: PLACEHOLDER_DATASET
  model_id: target-lora
  model_dir: PLACEHOLDER_MODEL
attack:
  method: recon
  num_samples: 8
  parameters:
    pretrained_model_name_or_path: runwayml/stable-diffusion-v1-5
    num_validation_images: 3
    inference_steps: 30
    eval_method: threshold
report:
  output_dir: experiments/recon-plan
"""


def create_minimal_recon_repo(repo_root: Path) -> None:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "inference.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "import torch\n"
        "\n"
        "# StableDiffusionPipeline\n"
        "\n"
        "def parse_args():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--pretrained_model_name_or_path')\n"
        "    parser.add_argument('--num_validation_images', type=int, default=3)\n"
        "    parser.add_argument('--inference', type=int, default=30)\n"
        "    parser.add_argument('--scheduler', default='default')\n"
        "    parser.add_argument('--output_dir')\n"
        "    parser.add_argument('--data_dir')\n"
        "    parser.add_argument('--save_dir')\n"
        "    return parser.parse_args()\n"
        "\n"
        "def main():\n"
        "    args = parse_args()\n"
        "    payload = torch.load(args.data_dir)\n"
        "    count = len(payload.get('text', []))\n"
        "    save_dir = Path(args.save_dir)\n"
        "    save_dir.mkdir(parents=True, exist_ok=True)\n"
        "    (save_dir / 'scheduler.txt').write_text(str(args.scheduler), encoding='utf-8')\n"
        "    for i in range(count):\n"
        "        for j in range(args.num_validation_images):\n"
        "            (save_dir / f'image_{i+1:02}_{j+1:02}.jpg').write_bytes(b'fake-jpg')\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    (repo_root / "cal_embedding.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "import torch\n"
        "\n"
        "# DeiTFeatureExtractor\n"
        "\n"
        "def parse_args():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--data_dir')\n"
        "    parser.add_argument('--sample_file')\n"
        "    parser.add_argument('--membership', type=int)\n"
        "    parser.add_argument('--img_num', type=int, default=3)\n"
        "    parser.add_argument('--gpu', type=int, default=0)\n"
        "    parser.add_argument('--save_dir')\n"
        "    parser.add_argument('--method', default='cosine')\n"
        "    parser.add_argument('--image_encoder', default='deit')\n"
        "    return parser.parse_args()\n"
        "\n"
        "def compute_scores():\n"
        "    return None\n"
        "\n"
        "def main():\n"
        "    args = parse_args()\n"
        "    payload = torch.load(args.data_dir)\n"
        "    count = len(payload.get('text', []))\n"
        "    score = 0.9 if int(args.membership) == 1 else 0.1\n"
        "    rows = [[ [score], int(args.membership) ] for _ in range(count)]\n"
        "    Path(args.save_dir).parent.mkdir(parents=True, exist_ok=True)\n"
        "    torch.save(rows, args.save_dir)\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    (repo_root / "test_accuracy.py").write_text(
        "import argparse\n"
        "import torch\n"
        "\n"
        "class DefineClassifier:\n    pass\n"
        "\n"
        "def parse_args():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--target_member_dir')\n"
        "    parser.add_argument('--target_non_member_dir')\n"
        "    parser.add_argument('--shadow_member_dir')\n"
        "    parser.add_argument('--shadow_non_member_dir')\n"
        "    parser.add_argument('--method', default='threshold')\n"
        "    return parser.parse_args()\n"
        "\n"
        "def process_data():\n"
        "    return None\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    args = parse_args()\n"
        "    target_member = torch.load(args.target_member_dir)\n"
        "    target_non_member = torch.load(args.target_non_member_dir)\n"
        "    print(f'Accuracy with Best Threshold: {0.75:.2f}')\n"
        "    print(f'AUC-ROC: {1.0:.3f}')\n",
        encoding="utf-8",
    )
    (repo_root / "train_text_to_image_lora.py").write_text("# train", encoding="utf-8")
    (repo_root / "kandinsky2_2_inference.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "import torch\n"
        "\n"
        "def parse_args():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--decoder_dir')\n"
        "    parser.add_argument('--prior_dir')\n"
        "    parser.add_argument('--save_dir')\n"
        "    parser.add_argument('--gpu', type=int, default=0)\n"
        "    parser.add_argument('--dataset_dir')\n"
        "    parser.add_argument('--num_validation_images', type=int, default=3)\n"
        "    parser.add_argument('--inference_steps', type=int, default=30)\n"
        "    return parser.parse_args()\n"
        "\n"
        "def main():\n"
        "    args = parse_args()\n"
        "    payload = torch.load(args.dataset_dir)\n"
        "    count = len(payload.get('text', []))\n"
        "    save_dir = Path(args.save_dir)\n"
        "    save_dir.mkdir(parents=True, exist_ok=True)\n"
        "    for i in range(count):\n"
        "        for j in range(args.num_validation_images):\n"
        "            (save_dir / f'image_{i+1:02}_{j+1:02}.jpg').write_bytes(b'fake-jpg')\n"
        "    (save_dir / 'inference_steps.txt').write_text(str(args.inference_steps), encoding='utf-8')\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )


def create_recon_dataset_payload(dataset_path: Path, count: int = 3) -> None:
    import torch

    payload = {
        "text": [f"prompt-{index}" for index in range(count)],
        "image": [f"image-{index}" for index in range(count)],
    }
    torch.save(payload, dataset_path)


def create_public_recon_bundle(bundle_root: Path, count: int = 3) -> None:
    import torch

    mapping = {
        "source-datasets/partial-100-target/member/dataset.pkl": ("member", count),
        "source-datasets/partial-100-target/non_member/dataset.pkl": ("non_member", count),
        "source-datasets/100-target/non_member/dataset.pkl": ("shadow_proxy", count),
        "source-datasets/100-shadow/non_member/dataset.pkl": ("shadow_non_member", count),
    }
    for relative_path, (prefix, sample_count) in mapping.items():
        path = bundle_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "text": [f"{prefix}-prompt-{index}" for index in range(sample_count)],
            "image": [f"{prefix}-image-{index}" for index in range(sample_count)],
        }
        torch.save(payload, path)


def load_kandinsky_inference_module(
    *,
    added_kv_processor_cls: type,
    processor_cls: type,
):
    global _KANDINSKY_INFERENCE_MODULE_CACHE
    if _KANDINSKY_INFERENCE_MODULE_CACHE is not None:
        return _KANDINSKY_INFERENCE_MODULE_CACHE

    script_path = (
        Path(__file__).resolve().parents[1]
        / "external"
        / "Reconstruction-based-Attack"
        / "kandinsky2_2_inference.py"
    )
    module_name = f"tests._kandinsky2_2_inference_stub_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load kandinsky2_2_inference.py for testing")

    diffusers_module = types.ModuleType("diffusers")
    diffusers_module.KandinskyV22Pipeline = object
    diffusers_module.KandinskyV22PriorPipeline = object

    diffusers_models_module = types.ModuleType("diffusers.models")
    diffusers_models_module.UNet2DConditionModel = object

    attention_module = types.ModuleType("diffusers.models.attention_processor")
    attention_module.LoRAAttnAddedKVProcessor = added_kv_processor_cls
    attention_module.LoRAAttnProcessor = processor_cls

    transformers_module = types.ModuleType("transformers")
    transformers_module.CLIPVisionModelWithProjection = object

    datasets_module = types.ModuleType("datasets")
    datasets_module.Dataset = object

    safetensors_module = types.ModuleType("safetensors")
    safetensors_torch_module = types.ModuleType("safetensors.torch")
    safetensors_torch_module.load_file = lambda path: {}

    stub_modules = {
        "diffusers": diffusers_module,
        "diffusers.models": diffusers_models_module,
        "diffusers.models.attention_processor": attention_module,
        "transformers": transformers_module,
        "datasets": datasets_module,
        "safetensors": safetensors_module,
        "safetensors.torch": safetensors_torch_module,
    }

    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stub_modules, clear=False):
        sys.modules.pop(module_name, None)
        spec.loader.exec_module(module)
    _KANDINSKY_INFERENCE_MODULE_CACHE = module
    return module


class ReconAttackTests(unittest.TestCase):
    def test_kandinsky_lora_helpers_support_zero_arg_processors(self) -> None:
        class ZeroArgAddedKVProcessor:
            def __init__(self) -> None:
                self.kind = "added"

        class ZeroArgProcessor:
            def __init__(self) -> None:
                self.kind = "base"

        module = load_kandinsky_inference_module(
            added_kv_processor_cls=ZeroArgAddedKVProcessor,
            processor_cls=ZeroArgProcessor,
        )

        added_processor = module.build_lora_added_kv_processor(
            hidden_size=320,
            cross_attention_dim=128,
            rank=4,
            device="cuda:0",
        )
        processor = module.build_lora_processor(hidden_size=320, device="cuda:0")

        self.assertIsInstance(added_processor, ZeroArgAddedKVProcessor)
        self.assertIsInstance(processor, ZeroArgProcessor)

    def test_kandinsky_prefers_native_attn_proc_loader_when_available(self) -> None:
        class ZeroArgAddedKVProcessor:
            def __init__(self) -> None:
                self.kind = "added"

        class ZeroArgProcessor:
            def __init__(self) -> None:
                self.kind = "base"

        class FakeModel:
            def __init__(self) -> None:
                self.loaded = None

            def load_attn_procs(self, state_dict) -> None:
                self.loaded = state_dict

        module = load_kandinsky_inference_module(
            added_kv_processor_cls=ZeroArgAddedKVProcessor,
            processor_cls=ZeroArgProcessor,
        )
        model = FakeModel()

        with patch.object(module, "load_weights", return_value={"lora": 1}) as mocked_load:
            loader_used = module.load_lora_attention_processors(model, "weights.safetensors")

        self.assertEqual(loader_used, "load_attn_procs")
        self.assertEqual(model.loaded, {"lora": 1})
        mocked_load.assert_called_once_with("weights.safetensors")

    def test_cli_plans_recon(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "query.pt"
            dataset_path.write_text("dataset", encoding="utf-8")
            model_dir = root / "lora"
            model_dir.mkdir()
            config_path = root / "audit.yaml"
            config_path.write_text(
                RECON_CONFIG_TEMPLATE
                .replace("PLACEHOLDER_DATASET", str(dataset_path).replace("\\", "/"))
                .replace("PLACEHOLDER_MODEL", str(model_dir).replace("\\", "/")),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["plan-recon", "--config", str(config_path)])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["inference_entrypoint"], "inference.py")
        self.assertEqual(payload["evaluation_entrypoint"], "test_accuracy.py")
        self.assertEqual(payload["inference_steps"], 30)

    def test_cli_probes_recon_assets(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "query.pt"
            dataset_path.write_text("dataset", encoding="utf-8")
            model_dir = root / "lora"
            model_dir.mkdir()
            config_path = root / "audit.yaml"
            config_path.write_text(
                RECON_CONFIG_TEMPLATE
                .replace("PLACEHOLDER_DATASET", str(dataset_path).replace("\\", "/"))
                .replace("PLACEHOLDER_MODEL", str(model_dir).replace("\\", "/")),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["probe-recon-assets", "--config", str(config_path)])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["checks"]["dataset_root"])
        self.assertTrue(payload["checks"]["model_dir"])

    def test_cli_runs_recon_dry_run(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "query.pt"
            dataset_path.write_text("dataset", encoding="utf-8")
            model_dir = root / "lora"
            model_dir.mkdir()
            repo_root = root / "recon"
            create_minimal_recon_repo(repo_root)

            config_path = root / "audit.yaml"
            config_path.write_text(
                RECON_CONFIG_TEMPLATE
                .replace("PLACEHOLDER_DATASET", str(dataset_path).replace("\\", "/"))
                .replace("PLACEHOLDER_MODEL", str(model_dir).replace("\\", "/")),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "dry-run-recon",
                        "--config",
                        str(config_path),
                        "--repo-root",
                        str(repo_root),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["checks"]["inference_has_pipeline"])
        self.assertTrue(payload["checks"]["evaluation_has_classifier"])

    def test_cli_runs_recon_eval_smoke(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "run-recon-eval-smoke",
                        "--workspace",
                        str(root / "recon-eval-smoke"),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["mode"], "eval-smoke")
        self.assertIn("auc", payload["metrics"])
        self.assertFalse((root / "recon-eval-smoke" / "synthetic-score-artifacts").exists())

    def test_summarize_recon_artifacts_writes_summary(self) -> None:
        from diffaudit.attacks.recon import summarize_recon_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_dir = root / "scores"
            artifact_dir.mkdir()
            torch_payloads = {
                "target_member.pt": [([0.91], 1), ([0.88], 1), ([0.86], 1)],
                "target_non_member.pt": [([0.18], 0), ([0.22], 0), ([0.25], 0)],
                "shadow_member.pt": [([0.89], 1), ([0.84], 1), ([0.83], 1)],
                "shadow_non_member.pt": [([0.17], 0), ([0.21], 0), ([0.24], 0)],
            }
            for filename, payload in torch_payloads.items():
                import torch

                torch.save(payload, artifact_dir / filename)

            result = summarize_recon_artifacts(
                artifact_dir=artifact_dir,
                workspace=root / "recon-artifact-summary",
            )

            self.assertTrue((root / "recon-artifact-summary" / "summary.json").exists())

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["mode"], "artifact-summary")
        self.assertIn("shadow_auc", result["metrics"])
        self.assertIn("target_auc", result["metrics"])

    def test_probe_recon_score_artifacts_reports_ready(self) -> None:
        from diffaudit.attacks.recon import probe_recon_score_artifacts
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_dir = root / "scores"
            artifact_dir.mkdir()
            torch_payloads = {
                "target_member.pt": [([0.91], 1), ([0.88], 1), ([0.86], 1)],
                "target_non_member.pt": [([0.18], 0), ([0.22], 0), ([0.25], 0)],
                "shadow_member.pt": [([0.89], 1), ([0.84], 1), ([0.83], 1)],
                "shadow_non_member.pt": [([0.17], 0), ([0.21], 0), ([0.24], 0)],
            }
            for filename, payload in torch_payloads.items():
                torch.save(payload, artifact_dir / filename)

            result = probe_recon_score_artifacts(artifact_dir)

        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["checks"]["all_required_files_present"])
        self.assertEqual(result["sample_counts"]["target_member"], 3)
        self.assertEqual(result["label_profiles"]["target_non_member"]["negative"], 3)

    def test_cli_summarizes_recon_artifacts(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_dir = root / "scores"
            artifact_dir.mkdir()
            torch_payloads = {
                "target_member.pt": [([0.91], 1), ([0.88], 1), ([0.86], 1)],
                "target_non_member.pt": [([0.18], 0), ([0.22], 0), ([0.25], 0)],
                "shadow_member.pt": [([0.89], 1), ([0.84], 1), ([0.83], 1)],
                "shadow_non_member.pt": [([0.17], 0), ([0.21], 0), ([0.24], 0)],
            }
            for filename, payload in torch_payloads.items():
                import torch

                torch.save(payload, artifact_dir / filename)

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "summarize-recon-artifacts",
                        "--artifact-dir",
                        str(artifact_dir),
                        "--workspace",
                        str(root / "recon-artifact-summary"),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["mode"], "artifact-summary")
        self.assertIn("target_auc", payload["metrics"])

    def test_cli_probes_recon_score_artifacts(self) -> None:
        from diffaudit.cli import main
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_dir = root / "scores"
            artifact_dir.mkdir()
            torch_payloads = {
                "target_member.pt": [([0.91], 1), ([0.88], 1), ([0.86], 1)],
                "target_non_member.pt": [([0.18], 0), ([0.22], 0), ([0.25], 0)],
                "shadow_member.pt": [([0.89], 1), ([0.84], 1), ([0.83], 1)],
                "shadow_non_member.pt": [([0.17], 0), ([0.21], 0), ([0.24], 0)],
            }
            for filename, payload in torch_payloads.items():
                torch.save(payload, artifact_dir / filename)

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "probe-recon-score-artifacts",
                        "--artifact-dir",
                        str(artifact_dir),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["sample_counts"]["shadow_member"], 3)

    def test_run_recon_upstream_eval_smoke_writes_summary(self) -> None:
        from diffaudit.attacks.recon import run_recon_upstream_eval_smoke

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "recon"
            create_minimal_recon_repo(repo_root)
            result = run_recon_upstream_eval_smoke(
                workspace=root / "recon-upstream-eval-smoke",
                repo_root=repo_root,
                method="threshold",
            )
            self.assertTrue((root / "recon-upstream-eval-smoke" / "summary.json").exists())
            self.assertFalse((root / "recon-upstream-eval-smoke" / "synthetic-score-artifacts").exists())

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["mode"], "upstream-eval-smoke")
        self.assertEqual(result["evaluation_method"], "threshold")
        self.assertTrue(result["checks"]["upstream_script_succeeded"])

    def test_cli_runs_recon_upstream_eval_smoke(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "recon"
            create_minimal_recon_repo(repo_root)

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "run-recon-upstream-eval-smoke",
                        "--workspace",
                        str(root / "recon-upstream-eval-smoke"),
                        "--repo-root",
                        str(repo_root),
                        "--method",
                        "threshold",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["mode"], "upstream-eval-smoke")
        self.assertTrue(payload["checks"]["upstream_script_succeeded"])

    def test_run_recon_mainline_smoke_writes_unified_summary(self) -> None:
        from diffaudit.attacks.recon import run_recon_mainline_smoke

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "recon"
            create_minimal_recon_repo(repo_root)

            result = run_recon_mainline_smoke(
                workspace=root / "recon-mainline-smoke",
                repo_root=repo_root,
                method="threshold",
            )

            workspace = root / "recon-mainline-smoke"
            self.assertTrue((workspace / "summary.json").exists())
            self.assertTrue((workspace / "eval-smoke" / "summary.json").exists())
            self.assertTrue((workspace / "artifact-summary" / "summary.json").exists())
            self.assertTrue((workspace / "upstream-eval-smoke" / "summary.json").exists())

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["mode"], "mainline-smoke")
        self.assertTrue(result["checks"]["all_stages_ready"])
        self.assertEqual(result["stages"]["artifact_summary"]["status"], "ready")
        self.assertEqual(result["stages"]["upstream_eval_smoke"]["status"], "ready")
        self.assertEqual(result["metrics"]["auc"], 1.0)

    def test_cli_runs_recon_mainline_smoke(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "recon"
            create_minimal_recon_repo(repo_root)

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "run-recon-mainline-smoke",
                        "--workspace",
                        str(root / "recon-mainline-smoke"),
                        "--repo-root",
                        str(repo_root),
                        "--method",
                        "threshold",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["mode"], "mainline-smoke")
        self.assertTrue(payload["checks"]["all_stages_ready"])
        self.assertEqual(payload["stages"]["eval_smoke"]["status"], "ready")

    def test_run_recon_artifact_mainline_writes_unified_summary(self) -> None:
        from diffaudit.attacks.recon import run_recon_artifact_mainline
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "recon"
            artifact_dir = root / "scores"
            create_minimal_recon_repo(repo_root)
            artifact_dir.mkdir()
            torch_payloads = {
                "target_member.pt": [([0.91], 1), ([0.88], 1), ([0.86], 1)],
                "target_non_member.pt": [([0.18], 0), ([0.22], 0), ([0.25], 0)],
                "shadow_member.pt": [([0.89], 1), ([0.84], 1), ([0.83], 1)],
                "shadow_non_member.pt": [([0.17], 0), ([0.21], 0), ([0.24], 0)],
            }
            for filename, payload in torch_payloads.items():
                torch.save(payload, artifact_dir / filename)

            result = run_recon_artifact_mainline(
                artifact_dir=artifact_dir,
                workspace=root / "recon-artifact-mainline",
                repo_root=repo_root,
                method="threshold",
            )

            workspace = root / "recon-artifact-mainline"
            self.assertTrue((workspace / "summary.json").exists())
            self.assertTrue((workspace / "artifact-summary" / "summary.json").exists())
            self.assertTrue((workspace / "upstream-eval" / "summary.json").exists())

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["mode"], "artifact-mainline")
        self.assertTrue(result["checks"]["all_stages_ready"])
        self.assertEqual(result["stages"]["artifact_summary"]["status"], "ready")
        self.assertEqual(result["stages"]["upstream_eval"]["status"], "ready")
        self.assertEqual(result["metrics"]["auc"], 1.0)

    def test_cli_runs_recon_artifact_mainline(self) -> None:
        from diffaudit.cli import main
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "recon"
            artifact_dir = root / "scores"
            create_minimal_recon_repo(repo_root)
            artifact_dir.mkdir()
            torch_payloads = {
                "target_member.pt": [([0.91], 1), ([0.88], 1), ([0.86], 1)],
                "target_non_member.pt": [([0.18], 0), ([0.22], 0), ([0.25], 0)],
                "shadow_member.pt": [([0.89], 1), ([0.84], 1), ([0.83], 1)],
                "shadow_non_member.pt": [([0.17], 0), ([0.21], 0), ([0.24], 0)],
            }
            for filename, payload in torch_payloads.items():
                torch.save(payload, artifact_dir / filename)

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "run-recon-artifact-mainline",
                        "--artifact-dir",
                        str(artifact_dir),
                        "--workspace",
                        str(root / "recon-artifact-mainline"),
                        "--repo-root",
                        str(repo_root),
                        "--method",
                        "threshold",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["mode"], "artifact-mainline")
        self.assertTrue(payload["checks"]["all_stages_ready"])
        self.assertEqual(payload["stages"]["upstream_eval"]["status"], "ready")

    def test_probe_recon_runtime_assets_reports_ready(self) -> None:
        from diffaudit.attacks.recon import probe_recon_runtime_assets

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "recon"
            create_minimal_recon_repo(repo_root)

            datasets = {}
            for name in (
                "target_member",
                "target_non_member",
                "shadow_member",
                "shadow_non_member",
            ):
                dataset_path = root / f"{name}.pt"
                create_recon_dataset_payload(dataset_path, count=2)
                datasets[name] = dataset_path

            target_model_dir = root / "target-lora"
            shadow_model_dir = root / "shadow-lora"
            target_model_dir.mkdir()
            shadow_model_dir.mkdir()

            result = probe_recon_runtime_assets(
                target_member_dataset=datasets["target_member"],
                target_nonmember_dataset=datasets["target_non_member"],
                shadow_member_dataset=datasets["shadow_member"],
                shadow_nonmember_dataset=datasets["shadow_non_member"],
                target_model_dir=target_model_dir,
                shadow_model_dir=shadow_model_dir,
                repo_root=repo_root,
            )

        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["checks"]["target_member_dataset"])
        self.assertTrue(result["checks"]["shadow_model_dir"])
        self.assertEqual(result["dataset_profiles"]["target_member"]["sample_count"], 2)
        self.assertTrue(result["dataset_profiles"]["target_member"]["has_text"])
        self.assertTrue(result["dataset_profiles"]["target_member"]["has_image"])

    def test_cli_runs_recon_runtime_mainline(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "recon"
            create_minimal_recon_repo(repo_root)

            datasets = {}
            for name in (
                "target_member",
                "target_non_member",
                "shadow_member",
                "shadow_non_member",
            ):
                dataset_path = root / f"{name}.pt"
                create_recon_dataset_payload(dataset_path, count=2)
                datasets[name] = dataset_path

            target_model_dir = root / "target-lora"
            shadow_model_dir = root / "shadow-lora"
            target_model_dir.mkdir()
            shadow_model_dir.mkdir()

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "run-recon-runtime-mainline",
                        "--target-member-dataset",
                        str(datasets["target_member"]),
                        "--target-nonmember-dataset",
                        str(datasets["target_non_member"]),
                        "--shadow-member-dataset",
                        str(datasets["shadow_member"]),
                        "--shadow-nonmember-dataset",
                        str(datasets["shadow_non_member"]),
                        "--target-model-dir",
                        str(target_model_dir),
                        "--shadow-model-dir",
                        str(shadow_model_dir),
                        "--workspace",
                        str(root / "recon-runtime-mainline"),
                        "--repo-root",
                        str(repo_root),
                        "--method",
                        "threshold",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["mode"], "runtime-mainline")
        self.assertTrue(payload["checks"]["runtime_assets_ready"])
        self.assertTrue(payload["checks"]["artifact_mainline_ready"])
        self.assertTrue(payload["checks"]["all_artifacts_generated"])
        self.assertEqual(payload["artifacts"]["target_member"]["sample_count"], 2)
        self.assertEqual(payload["stages"]["artifact_mainline"]["status"], "ready")

    def test_cli_runs_recon_runtime_mainline_with_ddim_scheduler(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "recon"
            create_minimal_recon_repo(repo_root)

            datasets = {}
            for name in (
                "target_member",
                "target_non_member",
                "shadow_member",
                "shadow_non_member",
            ):
                dataset_path = root / f"{name}.pt"
                create_recon_dataset_payload(dataset_path, count=1)
                datasets[name] = dataset_path

            target_model_dir = root / "target-lora"
            shadow_model_dir = root / "shadow-lora"
            target_model_dir.mkdir()
            shadow_model_dir.mkdir()

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "run-recon-runtime-mainline",
                        "--target-member-dataset",
                        str(datasets["target_member"]),
                        "--target-nonmember-dataset",
                        str(datasets["target_non_member"]),
                        "--shadow-member-dataset",
                        str(datasets["shadow_member"]),
                        "--shadow-nonmember-dataset",
                        str(datasets["shadow_non_member"]),
                        "--target-model-dir",
                        str(target_model_dir),
                        "--shadow-model-dir",
                        str(shadow_model_dir),
                        "--workspace",
                        str(root / "recon-runtime-mainline-ddim"),
                        "--repo-root",
                        str(repo_root),
                        "--scheduler",
                        "ddim",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["runtime"]["scheduler"], "ddim")
        self.assertEqual(payload["runtime"]["backend"], "stable_diffusion")
        self.assertIn("--scheduler", payload["artifacts"]["target_member"]["inference"]["command"])

    def test_run_recon_runtime_mainline_reuses_existing_score_artifacts(self) -> None:
        from diffaudit.attacks.recon import run_recon_runtime_mainline
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "recon"
            create_minimal_recon_repo(repo_root)

            datasets = {}
            for name in (
                "target_member",
                "target_non_member",
                "shadow_member",
                "shadow_non_member",
            ):
                dataset_path = root / f"{name}.pt"
                create_recon_dataset_payload(dataset_path, count=1)
                datasets[name] = dataset_path

            target_model_dir = root / "target-lora"
            shadow_model_dir = root / "shadow-lora"
            target_model_dir.mkdir()
            shadow_model_dir.mkdir()

            workspace = root / "recon-runtime-mainline-reuse"
            score_root = workspace / "score-artifacts"
            score_root.mkdir(parents=True, exist_ok=True)
            torch.save([([0.91], 1)], score_root / "target_member.pt")

            result = run_recon_runtime_mainline(
                target_member_dataset=datasets["target_member"],
                target_nonmember_dataset=datasets["target_non_member"],
                shadow_member_dataset=datasets["shadow_member"],
                shadow_nonmember_dataset=datasets["shadow_non_member"],
                target_model_dir=target_model_dir,
                shadow_model_dir=shadow_model_dir,
                workspace=workspace,
                repo_root=repo_root,
                method="threshold",
            )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["artifacts"]["target_member"]["status"], "ready")
        self.assertEqual(result["artifacts"]["target_member"]["stage"], "reused")
        self.assertTrue(result["checks"]["all_artifacts_generated"])

    def test_run_recon_runtime_mainline_regenerates_when_generated_images_incomplete(self) -> None:
        from diffaudit.attacks.recon import run_recon_runtime_mainline

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "recon"
            create_minimal_recon_repo(repo_root)
            (repo_root / "cal_embedding.py").write_text(
                "import argparse\n"
                "from pathlib import Path\n"
                "import torch\n"
                "\n"
                "def parse_args():\n"
                "    parser = argparse.ArgumentParser()\n"
                "    parser.add_argument('--data_dir')\n"
                "    parser.add_argument('--sample_file')\n"
                "    parser.add_argument('--membership', type=int)\n"
                "    parser.add_argument('--img_num', type=int, default=1)\n"
                "    parser.add_argument('--gpu', type=int, default=0)\n"
                "    parser.add_argument('--save_dir')\n"
                "    parser.add_argument('--method', default='cosine')\n"
                "    parser.add_argument('--image_encoder', default='deit')\n"
                "    return parser.parse_args()\n"
                "\n"
                "def main():\n"
                "    args = parse_args()\n"
                "    payload = torch.load(args.data_dir)\n"
                "    count = len(payload.get('text', []))\n"
                "    sample_dir = Path(args.sample_file)\n"
                "    for i in range(count):\n"
                "        expected = sample_dir / f'image_{i+1:02}_01.jpg'\n"
                "        if not expected.exists():\n"
                "            raise FileNotFoundError(expected)\n"
                "    score = 0.9 if int(args.membership) == 1 else 0.1\n"
                "    rows = [[[score], int(args.membership)] for _ in range(count)]\n"
                "    Path(args.save_dir).parent.mkdir(parents=True, exist_ok=True)\n"
                "    torch.save(rows, args.save_dir)\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    main()\n",
                encoding="utf-8",
            )

            datasets = {}
            for name in (
                "target_member",
                "target_non_member",
                "shadow_member",
                "shadow_non_member",
            ):
                dataset_path = root / f"{name}.pt"
                create_recon_dataset_payload(dataset_path, count=2)
                datasets[name] = dataset_path

            target_model_dir = root / "target-lora"
            shadow_model_dir = root / "shadow-lora"
            target_model_dir.mkdir()
            shadow_model_dir.mkdir()

            workspace = root / "recon-runtime-mainline-incomplete-generated"
            generated_dir = workspace / "generated-images" / "shadow_non_member"
            generated_dir.mkdir(parents=True, exist_ok=True)
            (generated_dir / "image_01_01.jpg").write_bytes(b"fake-jpg")

            result = run_recon_runtime_mainline(
                target_member_dataset=datasets["target_member"],
                target_nonmember_dataset=datasets["target_non_member"],
                shadow_member_dataset=datasets["shadow_member"],
                shadow_nonmember_dataset=datasets["shadow_non_member"],
                target_model_dir=target_model_dir,
                shadow_model_dir=shadow_model_dir,
                workspace=workspace,
                repo_root=repo_root,
                method="threshold",
                num_validation_images=1,
            )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["artifacts"]["shadow_non_member"]["status"], "ready")
        self.assertEqual(result["artifacts"]["shadow_non_member"]["sample_count"], 2)

    def test_cli_prepares_recon_public_subset(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_root = root / "bundle"
            create_public_recon_bundle(bundle_root, count=5)

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "prepare-recon-public-subset",
                        "--bundle-root",
                        str(bundle_root),
                        "--output-dir",
                        str(root / "prepared"),
                        "--target-count",
                        "2",
                        "--shadow-count",
                        "2",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["counts"]["target_member"], 2)
        self.assertEqual(payload["counts"]["shadow_member_proxy"], 2)
        self.assertTrue(payload["paths"]["mapping_note"].endswith("mapping-note.md"))

    def test_cli_audits_recon_public_bundle_semantics(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_root = root / "bundle"
            create_public_recon_bundle(bundle_root, count=5)
            for size in ("10", "25", "50", "100"):
                prepared_root = bundle_root / f"derived-public-{size}"
                payload = {
                    "status": "ready",
                    "output_dir": str(prepared_root),
                }
                prepared_root.mkdir(parents=True, exist_ok=True)
                (prepared_root / "mapping-note.md").write_text(
                    "\n".join(
                        [
                            "# Recon Public Subset Mapping",
                            "",
                            "- target_member <- source-datasets/partial-100-target/member/dataset.pkl",
                            "- target_non_member <- source-datasets/partial-100-target/non_member/dataset.pkl",
                            "- shadow_non_member <- source-datasets/100-shadow/non_member/dataset.pkl",
                            "- shadow_member_proxy <- source-datasets/100-target/non_member/dataset.pkl",
                            "",
                            "Note: shadow_member_proxy is an engineering proxy, not a fully validated paper-equivalent shadow-member split.",
                        ]
                    ),
                    encoding="utf-8",
                )
                _ = payload

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "audit-recon-public-bundle",
                        "--bundle-root",
                        str(bundle_root),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["semantic_gate"]["current_state"], "proxy-shadow-member")
        self.assertFalse(payload["semantic_gate"]["paper_aligned"])
        self.assertEqual(len(payload["derived_public_variants"]), 4)
        self.assertTrue(payload["checks"]["derived_public_mapping_notes_ready"])

    def test_cli_blocks_recon_stage0_when_bundle_is_proxy_semantic(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "recon"
            bundle_root = root / "bundle"
            create_minimal_recon_repo(repo_root)
            create_public_recon_bundle(bundle_root, count=5)
            prepared_root = bundle_root / "derived-public-100"
            prepared_root.mkdir(parents=True, exist_ok=True)
            (prepared_root / "mapping-note.md").write_text(
                "\n".join(
                    [
                        "# Recon Public Subset Mapping",
                        "",
                        "- target_member <- source-datasets/partial-100-target/member/dataset.pkl",
                        "- target_non_member <- source-datasets/partial-100-target/non_member/dataset.pkl",
                        "- shadow_non_member <- source-datasets/100-shadow/non_member/dataset.pkl",
                        "- shadow_member_proxy <- source-datasets/100-target/non_member/dataset.pkl",
                        "",
                        "Note: shadow_member_proxy is an engineering proxy, not a fully validated paper-equivalent shadow-member split.",
                    ]
                ),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "check-recon-stage0-paper-gate",
                        "--repo-root",
                        str(repo_root),
                        "--bundle-root",
                        str(bundle_root),
                        "--attack-scenario",
                        "attack-i",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["attack_scenario"], "attack-i")
        self.assertTrue(payload["checks"]["repo_root_exists"])
        self.assertTrue(payload["checks"]["bundle_root_exists"])
        self.assertTrue(payload["checks"]["local_semantic_chain_ready"])
        self.assertFalse(payload["checks"]["paper_aligned_semantics"])
        self.assertIn("paper_aligned_semantics", payload["missing_keys"])
        self.assertIn("proxy-shadow-member", payload["missing"])
        self.assertEqual(payload["allowed_claim"], "local-semantic-chain-ready")

    def test_cli_runs_recon_runtime_mainline_with_kandinsky_backend(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "recon"
            create_minimal_recon_repo(repo_root)

            datasets = {}
            for name in (
                "target_member",
                "target_non_member",
                "shadow_member",
                "shadow_non_member",
            ):
                dataset_path = root / f"{name}.pt"
                create_recon_dataset_payload(dataset_path, count=1)
                datasets[name] = dataset_path

            target_decoder = root / "target-decoder.pt"
            target_prior = root / "target-prior.pt"
            shadow_decoder = root / "shadow-decoder.pt"
            shadow_prior = root / "shadow-prior.pt"
            for path in (target_decoder, target_prior, shadow_decoder, shadow_prior):
                path.write_bytes(b"weights")

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "run-recon-runtime-mainline",
                        "--target-member-dataset",
                        str(datasets["target_member"]),
                        "--target-nonmember-dataset",
                        str(datasets["target_non_member"]),
                        "--shadow-member-dataset",
                        str(datasets["shadow_member"]),
                        "--shadow-nonmember-dataset",
                        str(datasets["shadow_non_member"]),
                        "--target-model-dir",
                        str(target_decoder),
                        "--shadow-model-dir",
                        str(shadow_decoder),
                        "--target-decoder-dir",
                        str(target_decoder),
                        "--target-prior-dir",
                        str(target_prior),
                        "--shadow-decoder-dir",
                        str(shadow_decoder),
                        "--shadow-prior-dir",
                        str(shadow_prior),
                        "--workspace",
                        str(root / "recon-runtime-mainline-kandinsky"),
                        "--repo-root",
                        str(repo_root),
                        "--backend",
                        "kandinsky_v22",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["runtime"]["backend"], "kandinsky_v22")
        self.assertIn("--decoder_dir", payload["artifacts"]["target_member"]["inference"]["command"])
        self.assertIn("--prior_dir", payload["artifacts"]["target_member"]["inference"]["command"])
        self.assertIn("--num_validation_images", payload["artifacts"]["target_member"]["inference"]["command"])
        self.assertIn("--inference_steps", payload["artifacts"]["target_member"]["inference"]["command"])


if __name__ == "__main__":
    unittest.main()
