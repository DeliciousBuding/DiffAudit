import json
import tempfile
import unittest
from pathlib import Path


class MoFitScaffoldTests(unittest.TestCase):
    def test_initialize_mofit_scaffold_creates_expected_artifacts(self) -> None:
        from diffaudit.attacks.mofit_scaffold import initialize_mofit_scaffold

        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "mofit-scaffold"
            result = initialize_mofit_scaffold(
                run_root=run_root,
                target_family="sd15-plus-celeba_partial_target-checkpoint-25000",
                caption_source="metadata-with-blip-fallback",
                surrogate_steps=8,
                embedding_steps=12,
                member_count=1,
                nonmember_count=1,
            )

            summary_path = run_root / "summary.json"
            records_path = run_root / "records.jsonl"
            surrogate_trace_dir = run_root / "traces" / "surrogate"
            embedding_trace_dir = run_root / "traces" / "embedding"

            self.assertEqual(result["summary_path"], summary_path.as_posix())
            self.assertTrue(summary_path.exists())
            self.assertTrue(records_path.exists())
            self.assertTrue(surrogate_trace_dir.exists())
            self.assertTrue(embedding_trace_dir.exists())
            self.assertEqual(records_path.read_text(encoding="utf-8"), "")

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["artifact_schema_version"], "mofit-interface-canary.v1")
            self.assertEqual(summary["status"], "scaffold_only")
            self.assertEqual(summary["target_family"], "sd15-plus-celeba_partial_target-checkpoint-25000")
            self.assertEqual(summary["caption_source"], "metadata-with-blip-fallback")
            self.assertEqual(summary["surrogate_steps"], 8)
            self.assertEqual(summary["embedding_steps"], 12)
            self.assertEqual(summary["member_count"], 1)
            self.assertEqual(summary["nonmember_count"], 1)


if __name__ == "__main__":
    unittest.main()
