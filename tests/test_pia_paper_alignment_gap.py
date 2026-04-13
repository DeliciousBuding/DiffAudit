import unittest
from pathlib import Path


class PIAAlignmentGapDocTests(unittest.TestCase):
    def test_pia_paper_alignment_gap_doc_exists_and_states_single_blocker(self) -> None:
        research_root = Path(__file__).resolve().parents[1]
        doc_path = research_root / "workspaces" / "gray-box" / "2026-04-09-pia-paper-alignment-gap.md"

        self.assertTrue(doc_path.exists(), f"missing document: {doc_path}")
        content = doc_path.read_text(encoding="utf-8")

        for phrase in (
            "workspace-verified",
            "paper-aligned",
            "checkpoint/source provenance",
            "唯一仍未闭环",
        ):
            self.assertIn(phrase, content)


if __name__ == "__main__":
    unittest.main()
