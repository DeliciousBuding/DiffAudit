import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


def create_minimal_clid_repo(repo_root: Path) -> None:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "mia_CLiD_impt.py").write_text(
        "from datasets import load_dataset\n"
        "flags.attack = 'clid_impt'\n"
        "from diffusers import AutoencoderKL\n",
        encoding="utf-8",
    )
    (repo_root / "cal_clid_th.py").write_text("# cal", encoding="utf-8")
    (repo_root / "cal_clid_xgb.py").write_text("# cal", encoding="utf-8")
    (repo_root / "train_text_to_image.py").write_text("# train", encoding="utf-8")


class ClidSmokeTests(unittest.TestCase):
    def test_run_clid_dry_run_smoke_writes_summary(self) -> None:
        from diffaudit.attacks.clid import run_clid_dry_run_smoke

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "CLiD"
            create_minimal_clid_repo(repo_root)

            result = run_clid_dry_run_smoke(
                workspace=root / "clid-dry-run-smoke",
                repo_root=repo_root,
            )

            self.assertTrue((root / "clid-dry-run-smoke" / "summary.json").exists())
            self.assertFalse((root / "clid-dry-run-smoke" / "synthetic-assets").exists())

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["mode"], "dry-run-smoke")
        self.assertTrue(result["checks"]["dry_run_ready"])

    def test_cli_runs_clid_dry_run_smoke(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "CLiD"
            create_minimal_clid_repo(repo_root)

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "run-clid-dry-run-smoke",
                        "--workspace",
                        str(root / "clid-dry-run-smoke"),
                        "--repo-root",
                        str(repo_root),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["mode"], "dry-run-smoke")
        self.assertTrue(payload["checks"]["dry_run_ready"])

    def test_summarize_clid_artifacts_writes_summary(self) -> None:
        from diffaudit.attacks.clid import summarize_clid_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_dir = root / "inter_output" / "CLID"
            artifact_dir.mkdir(parents=True)
            (artifact_dir / "Atk_Impt_M_coco_real_split1_DATA_val17_split1_TRTE_train_MAXsmp_3_T_test.txt").write_text(
                "0.10\t0.01\t0.01\t0.01\t0.01\n0.12\t0.02\t0.02\t0.02\t0.02\n",
                encoding="utf-8",
            )
            (artifact_dir / "Atk_Impt_M_coco_real_split1_DATA_val17_split1__TRTE_test_MAXsmp_3_T_test.txt").write_text(
                "0.80\t0.02\t0.02\t0.02\t0.02\n0.90\t0.03\t0.03\t0.03\t0.03\n",
                encoding="utf-8",
            )
            (artifact_dir / "Atk_Impt_M_coco_real_ori_DATA_val17_TRTE_train_MAXsmp_3_T_test.txt").write_text(
                "0.11\t0.01\t0.01\t0.01\t0.01\n0.14\t0.02\t0.02\t0.02\t0.02\n",
                encoding="utf-8",
            )
            (artifact_dir / "Atk_Impt_M_coco_real_ori_DATA_val17_TRTE_test_MAXsmp_3_T_test.txt").write_text(
                "0.75\t0.01\t0.01\t0.01\t0.01\n0.88\t0.02\t0.02\t0.02\t0.02\n",
                encoding="utf-8",
            )

            result = summarize_clid_artifacts(
                artifact_dir=artifact_dir,
                workspace=root / "clid-artifact-summary",
            )

            self.assertTrue((root / "clid-artifact-summary" / "summary.json").exists())

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["mode"], "artifact-summary")
        self.assertIn("shadow", result["metrics"])
        self.assertIn("target", result["metrics"])

    def test_cli_summarizes_clid_artifacts(self) -> None:
        from diffaudit.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_dir = root / "inter_output" / "CLID"
            artifact_dir.mkdir(parents=True)
            (artifact_dir / "Atk_Impt_M_coco_real_split1_DATA_val17_split1_TRTE_train_MAXsmp_3_T_test.txt").write_text(
                "0.10\t0.01\t0.01\t0.01\t0.01\n0.12\t0.02\t0.02\t0.02\t0.02\n",
                encoding="utf-8",
            )
            (artifact_dir / "Atk_Impt_M_coco_real_split1_DATA_val17_split1__TRTE_test_MAXsmp_3_T_test.txt").write_text(
                "0.80\t0.02\t0.02\t0.02\t0.02\n0.90\t0.03\t0.03\t0.03\t0.03\n",
                encoding="utf-8",
            )
            (artifact_dir / "Atk_Impt_M_coco_real_ori_DATA_val17_TRTE_train_MAXsmp_3_T_test.txt").write_text(
                "0.11\t0.01\t0.01\t0.01\t0.01\n0.14\t0.02\t0.02\t0.02\t0.02\n",
                encoding="utf-8",
            )
            (artifact_dir / "Atk_Impt_M_coco_real_ori_DATA_val17_TRTE_test_MAXsmp_3_T_test.txt").write_text(
                "0.75\t0.01\t0.01\t0.01\t0.01\n0.88\t0.02\t0.02\t0.02\t0.02\n",
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "summarize-clid-artifacts",
                        "--artifact-dir",
                        str(artifact_dir),
                        "--workspace",
                        str(root / "clid-artifact-summary"),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["mode"], "artifact-summary")
        self.assertIn("best_alpha", payload["metrics"]["shadow"])


if __name__ == "__main__":
    unittest.main()
