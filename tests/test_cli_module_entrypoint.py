import os
import subprocess
import sys
import unittest
from pathlib import Path


class CliModuleEntrypointTests(unittest.TestCase):
    def test_python_module_invocation_executes_main(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(repo_root / "src") if not existing else f"{repo_root / 'src'}{os.pathsep}{existing}"

        completed = subprocess.run(
            [sys.executable, "-m", "diffaudit.cli", "unsupported-command"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("invalid choice", completed.stderr)


if __name__ == "__main__":
    unittest.main()
