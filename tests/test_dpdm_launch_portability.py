from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"


def _read_script(name: str) -> str:
    return (SCRIPTS_ROOT / name).read_text(encoding="utf-8")


class TestDpdmLaunchPortability(unittest.TestCase):
    def test_launch_dpdm_training_exposes_overridable_paths(self) -> None:
        text = _read_script("launch_dpdm_training.ps1")
        self.assertIn("DIFFAUDIT_WORKSPACE_ROOT", text)
        self.assertIn("DIFFAUDIT_RESEARCH_PYTHON", text)
        self.assertNotRegex(text, r"[A-Z]:[/\\]Users[/\\][^/\\]+[/\\].*diffaudit-research[/\\]python\.exe")
        self.assertNotRegex(text, r"[A-Z]:[/\\]Code[/\\]DiffAudit[/\\]Research[/\\]external[/\\]DPDM")

    def test_target_and_shadow_launchers_do_not_require_repo_author_paths(self) -> None:
        for name in (
            "launch_dpdm_target_and_shadows.ps1",
            "launch_dpdm_shadow_sequence.ps1",
            "launch_dpdm_shadow_sequence_after_pid.ps1",
        ):
            text = _read_script(name)
            self.assertIn("$PSScriptRoot", text, msg=name)
            self.assertNotRegex(text, r"[A-Z]:[/\\]Code[/\\]DiffAudit[/\\]Research[/\\]scripts", msg=name)

    def test_scripts_readme_marks_scheduler_optional(self) -> None:
        text = (SCRIPTS_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("scheduler", text.lower())
        self.assertIn("可选", text)
        self.assertIn("DIFFAUDIT_WORKSPACE_ROOT", text)

    def test_launchers_skip_empty_optional_forwarded_paths(self) -> None:
        for name in (
            "launch_dpdm_target_and_shadows.ps1",
            "launch_dpdm_shadow_sequence.ps1",
            "launch_dpdm_shadow_sequence_after_pid.ps1",
        ):
            text = _read_script(name)
            self.assertIn("IsNullOrWhiteSpace($DpdmRoot)", text, msg=name)
            self.assertIn("IsNullOrWhiteSpace($ConfigPath)", text, msg=name)

    def test_shadow_sequence_defaults_to_three_shadows(self) -> None:
        text = _read_script("launch_dpdm_shadow_sequence.ps1")
        self.assertIn('@("shadow-01", "shadow-02", "shadow-03")', text)


if __name__ == "__main__":
    unittest.main()
