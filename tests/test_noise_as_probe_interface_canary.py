import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_script_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "run_noise_as_probe_interface_canary.py"
    spec = importlib.util.spec_from_file_location("run_noise_as_probe_interface_canary", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NoiseAsProbeInterfaceCanaryTests(unittest.TestCase):
    def test_load_rows_returns_empty_for_zero_limit(self) -> None:
        module = _load_script_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "member"
            split_dir.mkdir(parents=True, exist_ok=True)
            (split_dir / "metadata.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"file_name": "a.png", "text": "a"}),
                        json.dumps({"file_name": "b.png", "text": "b"}),
                    ]
                ),
                encoding="utf-8",
            )

            rows = module.load_rows(split_dir, subset_limit=0, offset=0)

            self.assertEqual(rows, [])

    def test_resolve_generation_guidance_config_fixed_defaults_to_single_scale(self) -> None:
        module = _load_script_module()

        args = argparse.Namespace(
            generation_guidance_mode="fixed",
            generation_guidance_scale=3.5,
            generation_guidance_scales=None,
            generation_guidance_seed=0,
        )

        config = module.resolve_generation_guidance_config(args)

        self.assertEqual(config["mode"], "fixed")
        self.assertEqual(config["scales"], [3.5])
        self.assertEqual(config["seed"], 0)

    def test_resolve_generation_guidance_config_hidden_jitter_requires_multiple_scales(self) -> None:
        module = _load_script_module()

        args = argparse.Namespace(
            generation_guidance_mode="hidden-jitter",
            generation_guidance_scale=3.5,
            generation_guidance_scales="3.5",
            generation_guidance_seed=0,
        )

        with self.assertRaises(ValueError):
            module.resolve_generation_guidance_config(args)

    def test_deterministic_guidance_choice_is_stable_for_same_sample(self) -> None:
        module = _load_script_module()

        first = module.deterministic_guidance_choice(
            split_name="member_001",
            file_name="065.png",
            scales=[3.5, 7.5],
            seed=7,
        )
        second = module.deterministic_guidance_choice(
            split_name="member_001",
            file_name="065.png",
            scales=[3.5, 7.5],
            seed=7,
        )

        self.assertEqual(first, second)
        self.assertEqual(first, (1, 7.5))

    def test_summarize_guidance_usage_groups_by_effective_split(self) -> None:
        module = _load_script_module()

        usage = module.summarize_guidance_usage(
            [
                {"split": "member_000", "generation_guidance_scale_used": 3.5},
                {"split": "member_001", "generation_guidance_scale_used": 7.5},
                {"split": "nonmember_000", "generation_guidance_scale_used": 3.5},
                {"split": "calibration_nonmember_000", "generation_guidance_scale_used": 7.5},
            ]
        )

        self.assertEqual(usage["all"], {"3.5": 2, "7.5": 2})
        self.assertEqual(usage["member"], {"3.5": 1, "7.5": 1})
        self.assertEqual(usage["nonmember"], {"3.5": 1})
        self.assertEqual(usage["calibration_nonmember"], {"7.5": 1})

    def test_ensure_disjoint_rows_rejects_overlap(self) -> None:
        module = _load_script_module()

        with self.assertRaises(ValueError):
            module.ensure_disjoint_rows(
                [{"file_name": "080.png"}, {"file_name": "081.png"}],
                [{"file_name": "081.png"}],
                primary_name="evaluation nonmember rows",
                calibration_name="calibration rows",
            )


if __name__ == "__main__":
    unittest.main()
