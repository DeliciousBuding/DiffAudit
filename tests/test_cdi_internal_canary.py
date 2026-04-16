import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_script_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "run_cdi_internal_canary.py"
    spec = importlib.util.spec_from_file_location("run_cdi_internal_canary", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CdiInternalCanaryTests(unittest.TestCase):
    def test_build_collection_split_is_deterministic(self) -> None:
        module = _load_script_module()

        split = module.build_collection_split(
            member_count=1024,
            nonmember_count=1024,
            control_size=4,
            test_size=4,
        )

        self.assertEqual(split["P_ctrl"], [0, 1, 2, 3])
        self.assertEqual(split["P_test"], [4, 5, 6, 7])
        self.assertEqual(split["U_ctrl"], [0, 1, 2, 3])
        self.assertEqual(split["U_test"], [4, 5, 6, 7])

    def test_build_collection_split_rejects_small_payloads(self) -> None:
        module = _load_script_module()

        with self.assertRaises(ValueError):
            module.build_collection_split(
                member_count=3,
                nonmember_count=8,
                control_size=2,
                test_size=2,
            )

    def test_run_internal_canary_emits_secmi_only_artifacts(self) -> None:
        module = _load_script_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            secmi_scores = root / "secmi_scores.json"
            secmi_scores.write_text(
                json.dumps(
                    {
                        "member_scores": [0.9, 0.8, 0.7, 0.6],
                        "nonmember_scores": [0.1, 0.2, 0.3, 0.4],
                    }
                ),
                encoding="utf-8",
            )

            args = module.argparse.Namespace(
                run_root=root / "run",
                secmi_scores=secmi_scores,
                pia_scores=None,
                control_size=2,
                test_size=2,
                resamples=1,
                seed=0,
            )

            summary = module.run_internal_canary(args)

            self.assertEqual(summary["feature_mode"], "secmi-stat-only")
            self.assertEqual(summary["collection_counts"]["P_ctrl"], 2)
            self.assertEqual(summary["analysis"]["secmi_memberness_orientation"], "identity")
            self.assertAlmostEqual(summary["metrics"]["secmi_p_test_mean"], 0.65)
            self.assertAlmostEqual(summary["metrics"]["secmi_u_test_mean"], 0.35)

            collections = json.loads((args.run_root / "collections.json").read_text(encoding="utf-8"))
            self.assertEqual(collections["collections"]["P_ctrl"], [0, 1])

            rows = [
                json.loads(line)
                for line in (args.run_root / "sample_scores.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 8)
            self.assertIn("secmi_stat_score", rows[0])
            self.assertNotIn("pia_score", rows[0])

            audit_summary = json.loads((args.run_root / "audit_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(audit_summary["feature_mode"], "secmi-stat-only")

    def test_run_internal_canary_emits_paired_scores_when_available(self) -> None:
        module = _load_script_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            secmi_scores = root / "secmi_scores.json"
            pia_scores = root / "pia_scores.json"
            secmi_scores.write_text(
                json.dumps(
                    {
                        "member_scores": [0.9, 0.8, 0.7, 0.6],
                        "nonmember_scores": [0.1, 0.2, 0.3, 0.4],
                        "member_indices": [10, 11, 12, 13],
                        "nonmember_indices": [20, 21, 22, 23],
                    }
                ),
                encoding="utf-8",
            )
            pia_scores.write_text(
                json.dumps(
                    {
                        "member_scores": [10.0, 9.0, 8.0, 7.0],
                        "nonmember_scores": [1.0, 2.0, 3.0, 4.0],
                        "member_indices": [10, 11, 12, 13],
                        "nonmember_indices": [20, 21, 22, 23],
                    }
                ),
                encoding="utf-8",
            )

            args = module.argparse.Namespace(
                run_root=root / "run",
                secmi_scores=secmi_scores,
                pia_scores=pia_scores,
                control_size=2,
                test_size=2,
                resamples=1,
                seed=0,
            )

            summary = module.run_internal_canary(args)

            self.assertEqual(summary["feature_mode"], "paired-pia-secmi")
            self.assertAlmostEqual(summary["metrics"]["pia_p_test_mean"], 7.5)
            self.assertAlmostEqual(summary["metrics"]["pia_u_test_mean"], 3.5)

            rows = [
                json.loads(line)
                for line in (args.run_root / "sample_scores.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertIn("pia_score", rows[0])
            self.assertEqual(rows[0]["source_index"], 10)

    def test_orient_memberness_negates_lower_is_more_member_like_scores(self) -> None:
        module = _load_script_module()

        oriented = module.orient_memberness(
            {
                "member_scores": [0.1, 0.2],
                "nonmember_scores": [0.8, 0.9],
                "member_indices": [0, 1],
                "nonmember_indices": [2, 3],
            }
        )

        self.assertEqual(oriented["memberness_orientation"], "negated")
        self.assertGreater(sum(oriented["member_scores"]) / 2.0, sum(oriented["nonmember_scores"]) / 2.0)


if __name__ == "__main__":
    unittest.main()
