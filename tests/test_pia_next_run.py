import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PiaNextRunTests(unittest.TestCase):
    def test_pia_next_run_emits_manifest_and_provenance(self) -> None:
        research_root = Path(__file__).resolve().parents[1]
        tool_src = research_root / "tools" / "pia_next_run" / "src"

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "repo"
            repo_root.mkdir()
            subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
            (repo_root / "README.md").write_text("repo\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=repo_root, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo_root, check=True, capture_output=True)

            config_path = root / "pia.yaml"
            config_path.write_text("task:\n  name: pia-next-run\n", encoding="utf-8")

            member_split_root = root / "member_splits"
            member_split_root.mkdir()
            (member_split_root / "CIFAR10_train_ratio0.5.npz").write_bytes(b"split")

            out_dir = root / "out"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(tool_src)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pia_next_run",
                    "--config",
                    str(config_path),
                    "--repo-root",
                    str(repo_root),
                    "--member-split-root",
                    str(member_split_root),
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=research_root,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            manifest_path = out_dir / "manifest.json"
            provenance_path = out_dir / "provenance.json"
            self.assertTrue(manifest_path.exists())
            self.assertTrue(provenance_path.exists())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "pia_next_run.manifest.v1")
            self.assertTrue(manifest["validation"]["ok"])
            self.assertEqual(
                manifest["paths"]["member_split_root"],
                str(member_split_root),
            )


if __name__ == "__main__":
    unittest.main()
