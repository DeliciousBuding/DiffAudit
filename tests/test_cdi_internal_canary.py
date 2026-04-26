import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


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
                secmi_run_root=None,
                pia_scores=None,
                paired_scorer="auto",
                control_size=2,
                test_size=2,
                resamples=1,
                seed=0,
            )

            summary = module.run_internal_canary(args)

            self.assertEqual(summary["feature_mode"], "secmi-stat-only")
            self.assertEqual(summary["contract"]["paired_scorer_policy_effective"], "none")
            self.assertFalse(summary["contract"]["component_reporting_required"])
            self.assertEqual(summary["collection_counts"]["P_ctrl"], 2)
            self.assertEqual(summary["analysis"]["secmi_memberness_orientation"], "identity")
            self.assertEqual(summary["analysis"]["paired_scorer_policy_effective"], "none")
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
                secmi_run_root=None,
                pia_scores=pia_scores,
                paired_scorer="none",
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

    def test_run_internal_canary_auto_policy_enables_default_paired_scorer(self) -> None:
        module = _load_script_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            secmi_scores = root / "secmi_scores.json"
            pia_scores = root / "pia_scores.json"
            secmi_scores.write_text(
                json.dumps(
                    {
                        "member_scores": [0.95, 0.85, 0.75, 0.65],
                        "nonmember_scores": [0.05, 0.15, 0.25, 0.35],
                    }
                ),
                encoding="utf-8",
            )
            pia_scores.write_text(
                json.dumps(
                    {
                        "member_scores": [10.0, 9.0, 8.0, 7.0],
                        "nonmember_scores": [1.0, 2.0, 3.0, 4.0],
                    }
                ),
                encoding="utf-8",
            )

            args = module.argparse.Namespace(
                run_root=root / "run",
                secmi_scores=secmi_scores,
                secmi_run_root=None,
                pia_scores=pia_scores,
                paired_scorer="auto",
                control_size=2,
                test_size=2,
                resamples=1,
                seed=0,
            )

            summary = module.run_internal_canary(args)

            self.assertEqual(summary["feature_mode"], "paired-pia-secmi-control-z-linear")
            self.assertEqual(summary["analysis"]["paired_scorer_policy_requested"], "auto")
            self.assertEqual(summary["analysis"]["paired_scorer_policy_effective"], "control-z-linear")
            self.assertEqual(summary["contract"]["paired_scorer_policy_effective"], "control-z-linear")
            self.assertTrue(summary["contract"]["component_reporting_required"])
            self.assertIn("paired_p_test_mean", summary["metrics"])

    def test_run_internal_canary_emits_control_z_linear_paired_scores(self) -> None:
        module = _load_script_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            secmi_scores = root / "secmi_scores.json"
            pia_scores = root / "pia_scores.json"
            secmi_scores.write_text(
                json.dumps(
                    {
                        "member_scores": [0.95, 0.85, 0.75, 0.65],
                        "nonmember_scores": [0.05, 0.15, 0.25, 0.35],
                    }
                ),
                encoding="utf-8",
            )
            pia_scores.write_text(
                json.dumps(
                    {
                        "member_scores": [10.0, 9.0, 8.0, 7.0],
                        "nonmember_scores": [1.0, 2.0, 3.0, 4.0],
                    }
                ),
                encoding="utf-8",
            )

            args = module.argparse.Namespace(
                run_root=root / "run",
                secmi_scores=secmi_scores,
                secmi_run_root=None,
                pia_scores=pia_scores,
                paired_scorer="control-z-linear",
                control_size=2,
                test_size=2,
                resamples=1,
                seed=0,
            )

            summary = module.run_internal_canary(args)

            self.assertEqual(summary["feature_mode"], "paired-pia-secmi-control-z-linear")
            self.assertIn("paired_p_test_mean", summary["metrics"])
            self.assertIn("paired_u_test_mean", summary["metrics"])
            self.assertGreater(summary["metrics"]["paired_p_test_mean"], summary["metrics"]["paired_u_test_mean"])
            weights = summary["analysis"]["paired_scorer_details"]["weights"]
            self.assertAlmostEqual(sum(weights.values()), 1.0)

            rows = [
                json.loads(line)
                for line in (args.run_root / "sample_scores.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertIn("paired_score", rows[0])
            self.assertIn("secmi_z_score", rows[0])
            self.assertIn("pia_z_score", rows[0])

    def test_load_score_payload_from_run_root_reads_npy_artifacts(self) -> None:
        module = _load_script_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            np.save(run_root / "secmi_member_scores.npy", np.asarray([0.9, 0.8], dtype=float))
            np.save(run_root / "secmi_nonmember_scores.npy", np.asarray([0.1, 0.2], dtype=float))

            payload = module.load_score_payload_from_run_root(run_root)
            self.assertEqual(payload["member_scores"], [0.9, 0.8])
            self.assertEqual(payload["nonmember_scores"], [0.1, 0.2])
            self.assertIsNone(payload["member_indices"])

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
