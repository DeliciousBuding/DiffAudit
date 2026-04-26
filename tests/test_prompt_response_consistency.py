import tempfile
import unittest
from pathlib import Path

import numpy as np


class PromptResponseConsistencyTests(unittest.TestCase):
    def test_collect_generated_paths_matches_expected_prefix(self) -> None:
        from diffaudit.attacks.prompt_response_consistency import collect_generated_paths

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "000_foo_ret0.png").write_text("x", encoding="utf-8")
            (root / "000_foo_ret1.png").write_text("x", encoding="utf-8")
            (root / "001_bar_ret0.png").write_text("x", encoding="utf-8")

            paths = collect_generated_paths(root, sample_index=0, file_name="foo.png")

            self.assertEqual([path.name for path in paths], ["000_foo_ret0.png", "000_foo_ret1.png"])

    def test_summarize_prompt_response_reports_gain_against_mean_cos(self) -> None:
        from diffaudit.attacks.prompt_response_consistency import summarize_prompt_response

        labels = np.asarray([1, 1, 0, 0], dtype=int)
        scores = np.asarray([0.95, 0.85, 0.35, 0.25], dtype=float)
        baseline = np.asarray([0.8, 0.3, 0.7, 0.2], dtype=float)

        result = summarize_prompt_response(labels, scores, baseline)

        self.assertGreater(result["auc"], 0.9)
        self.assertGreater(result["auc_gain_vs_mean_cos"], 0.0)


if __name__ == "__main__":
    unittest.main()
