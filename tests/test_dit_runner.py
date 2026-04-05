import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


def create_minimal_dit_repo(repo_root: Path) -> None:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "download.py").write_text(
        "def find_model(model_name):\n    return {'ok': True}\n",
        encoding="utf-8",
    )
    (repo_root / "models.py").write_text(
        "DiT_models = {'DiT-XL/2': lambda **kwargs: None}\n",
        encoding="utf-8",
    )
    (repo_root / "sample.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--model', default='DiT-XL/2')\n"
        "    parser.add_argument('--image-size', type=int, default=256)\n"
        "    parser.add_argument('--num-sampling-steps', type=int, default=250)\n"
        "    parser.add_argument('--seed', type=int, default=0)\n"
        "    parser.add_argument('--ckpt', default=None)\n"
        "    parser.add_argument('--save-path', default='sample.png')\n"
        "    args = parser.parse_args()\n"
        "    Path(args.save_path).write_bytes(b'fake-png')\n"
        "    print(f'SAVED:{args.save_path}')\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    (repo_root / "diffusion").mkdir(exist_ok=True)


class DiTRunnerTests(unittest.TestCase):
    def test_probe_dit_assets_reports_ready(self) -> None:
        from diffaudit.attacks.dit import probe_dit_assets

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "DiT"
            create_minimal_dit_repo(repo_root)
            result = probe_dit_assets(repo_root=repo_root, ckpt=None, model="DiT-XL/2", image_size=256)

        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["checks"]["workspace_files"])
        self.assertTrue(result["checks"]["autodownload_allowed"])

    def test_cli_runs_dit_sample_smoke(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "DiT"
            create_minimal_dit_repo(repo_root)

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "run-dit-sample-smoke",
                        "--workspace",
                        str(root / "dit-sample-smoke"),
                        "--repo-root",
                        str(repo_root),
                        "--model",
                        "DiT-XL/2",
                        "--image-size",
                        "256",
                        "--num-sampling-steps",
                        "2",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["mode"], "sample-smoke")
        self.assertTrue(payload["checks"]["sample_script_succeeded"])
        self.assertTrue(payload["checks"]["sample_image_created"])


if __name__ == "__main__":
    unittest.main()
