import tempfile
import unittest
from pathlib import Path


class ResearchAutomationHealthTests(unittest.TestCase):
    def test_classify_run_summary_mention_distinguishes_placeholder_legacy_and_active(self) -> None:
        from diffaudit.research_automation_health import classify_run_summary_mention

        self.assertEqual(
            classify_run_summary_mention("AGENTS.md", "workspaces/<lane>/runs/<run-name>/summary.json"),
            "template_placeholder",
        )
        self.assertEqual(
            classify_run_summary_mention("legacy/old.md", "workspaces/black-box/runs/real/summary.json"),
            "legacy_reference",
        )
        self.assertEqual(
            classify_run_summary_mention("docs/reproduction-status.md", "workspaces/black-box/runs/real/summary.json"),
            "active_reference",
        )

    def test_extract_priority_ladder_groups_sections(self) -> None:
        from diffaudit.research_automation_health import extract_priority_ladder

        roadmap_text = """
### Top now
1. ⬜ `GB-1` second gray-box defense
2. ✅ `X-3` system sync

### Next
6. ⬜ `INF-2` research automation health

### Then
10. ⬜ `INF-3` subagent leverage experiments
"""
        result = extract_priority_ladder(roadmap_text)

        self.assertEqual(result["Top now"][0], "1. ⬜ `GB-1` second gray-box defense")
        self.assertEqual(result["Next"][0], "6. ⬜ `INF-2` research automation health")
        self.assertEqual(result["Then"][0], "10. ⬜ `INF-3` subagent leverage experiments")

    def test_audit_marks_only_actionable_ignored_run_summary_as_friction(self) -> None:
        from diffaudit.research_automation_health import audit_research_automation_health

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".gitignore").write_text("workspaces/black-box/runs/\n", encoding="utf-8")
            (root / "ROADMAP.md").write_text(
                "\n".join(
                    [
                        "### Top now",
                        "1. ⬜ `GB-1` second gray-box defense",
                        "### Next",
                        "6. ⬜ `INF-2` research automation health",
                        "### Then",
                        "10. ⬜ `INF-3` subagent leverage experiments",
                        "- canonical evidence anchor:",
                        "  - `workspaces/black-box/runs/test-run/summary.json`",
                    ]
                ),
                encoding="utf-8",
            )
            (root / "AGENTS.md").write_text(
                "- `workspaces/<lane>/runs/<run-name>/summary.json`\n",
                encoding="utf-8",
            )
            summary_path = root / "workspaces" / "black-box" / "runs" / "test-run" / "summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text("{}", encoding="utf-8")
            (root / "workspaces" / "black-box" / "note.md").write_text(
                "- `next GPU candidate`: `test`\n- `active_gpu_question`: `none`\n",
                encoding="utf-8",
            )

            import subprocess

            subprocess.run(["git", "-C", str(root), "init"], check=True, capture_output=True)

            result = audit_research_automation_health(root)

            self.assertEqual(result["counts"]["gpu_candidate_markers"], 1)
            self.assertEqual(result["counts"]["idle_reason_markers"], 1)
            self.assertEqual(result["counts"]["template_placeholder_run_summaries"], 1)
            self.assertEqual(result["counts"]["actionable_ignored_run_summaries"], 1)
            self.assertIn(
                "some active run-summary anchors point to ignored files and need force-add discipline",
                result["friction_points"],
            )

    def test_audit_marks_actionable_untracked_run_summary_as_friction(self) -> None:
        from diffaudit.research_automation_health import audit_research_automation_health

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "ROADMAP.md").write_text(
                "\n".join(
                    [
                        "### Top now",
                        "1. ⬜ `GB-1` second gray-box defense",
                        "### Next",
                        "6. ⬜ `INF-2` research automation health",
                        "### Then",
                        "10. ⬜ `INF-3` subagent leverage experiments",
                        "- canonical evidence anchor:",
                        "  - `workspaces/black-box/runs/test-run/summary.json`",
                    ]
                ),
                encoding="utf-8",
            )
            summary_path = root / "workspaces" / "black-box" / "runs" / "test-run" / "summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text("{}", encoding="utf-8")
            (root / "workspaces" / "black-box" / "note.md").write_text(
                "- `next GPU candidate`: `test`\n- `active_gpu_question`: `none`\n",
                encoding="utf-8",
            )

            import subprocess

            subprocess.run(["git", "-C", str(root), "init"], check=True, capture_output=True)

            result = audit_research_automation_health(root)

            self.assertEqual(result["counts"]["untracked_run_summaries"], 1)
            self.assertEqual(result["counts"]["actionable_untracked_run_summaries"], 1)
            self.assertEqual(result["status"], "friction detected")
            self.assertIn(
                "some active run-summary anchors point to untracked files and need add discipline",
                result["friction_points"],
            )

    def test_audit_excludes_legacy_run_summary_mentions_from_actionable_friction(self) -> None:
        from diffaudit.research_automation_health import audit_research_automation_health

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".gitignore").write_text("workspaces/black-box/runs/\n", encoding="utf-8")
            (root / "ROADMAP.md").write_text(
                "\n".join(
                    [
                        "### Top now",
                        "1. ⬜ `GB-1` second gray-box defense",
                        "### Next",
                        "6. ⬜ `INF-2` research automation health",
                        "### Then",
                        "10. ⬜ `INF-3` subagent leverage experiments",
                    ]
                ),
                encoding="utf-8",
            )
            summary_path = root / "workspaces" / "black-box" / "runs" / "test-run" / "summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text("{}", encoding="utf-8")
            legacy_dir = root / "legacy"
            legacy_dir.mkdir()
            (legacy_dir / "old-note.md").write_text(
                "- `workspaces/black-box/runs/test-run/summary.json`\n",
                encoding="utf-8",
            )
            (root / "workspaces" / "black-box" / "note.md").write_text(
                "- `next GPU candidate`: `test`\n- `active_gpu_question`: `none`\n",
                encoding="utf-8",
            )

            import subprocess

            subprocess.run(["git", "-C", str(root), "init"], check=True, capture_output=True)

            result = audit_research_automation_health(root)

            self.assertEqual(result["counts"]["legacy_run_summaries"], 1)
            self.assertEqual(result["counts"]["actionable_run_summaries"], 0)
            self.assertNotIn(
                "some active run-summary anchors point to ignored files and need force-add discipline",
                result["friction_points"],
            )


if __name__ == "__main__":
    unittest.main()
