import json
import tempfile
import unittest
from pathlib import Path


class ClidThresholdAuditTests(unittest.TestCase):
    def test_audit_clid_threshold_compatibility_detects_target_ready_shadow_block(self) -> None:
        from diffaudit.attacks.clid_threshold_audit import audit_clid_threshold_compatibility

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_run = root / "clid-target-run"
            outputs = source_run / "outputs"
            outputs.mkdir(parents=True)

            header = (
                "{'diff_path': "
                "'C:/Users/exampleuser/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/x'}\t"
                "C:/Users/exampleuser/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/x"
            )
            train_path = outputs / "Atk_clid_clip_M_local_target_TRTE_train.txt"
            test_path = outputs / "Atk_clid_clip_M_local_target_TRTE_test.txt"
            train_path.write_text(
                "\n".join(
                    [
                        header,
                        "0.1\t0.01\t0.02\t0.03\t0.04",
                        "0.2\t0.02\t0.03\t0.04\t0.05",
                    ]
                ),
                encoding="utf-8",
            )
            test_path.write_text(
                "\n".join(
                    [
                        header,
                        "0.3\t0.03\t0.04\t0.05\t0.06",
                        "0.4\t0.04\t0.05\t0.06\t0.07",
                    ]
                ),
                encoding="utf-8",
            )

            prepared_run = root / "clid-prepared-run"
            prepared_run.mkdir()
            (prepared_run / "config.json").write_text(
                json.dumps(
                    {
                        "assets": {
                            "base_model": "C:/DiffAudit/Download/shared/weights/stable-diffusion-v1-5"
                        }
                    }
                ),
                encoding="utf-8",
            )

            workspace = root / "clid-threshold-audit"
            result = audit_clid_threshold_compatibility(
                source_run=source_run,
                workspace=workspace,
                prepared_run=prepared_run,
            )

            self.assertEqual(result["status"], "ready")
            self.assertTrue(result["target_pair"]["ready"])
            self.assertEqual(result["target_pair"]["train_shape"], [2, 5])
            self.assertEqual(result["target_pair"]["test_shape"], [2, 5])
            self.assertFalse(result["shadow_pair"]["ready"])
            self.assertTrue(result["boundary_findings"]["executed_run_cache_root_leak"])
            self.assertTrue(result["boundary_findings"]["prepared_run_uses_staged_root"])
            self.assertEqual(result["verdict"], "evaluator-near but shadow-blocked")
            self.assertTrue((workspace / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
