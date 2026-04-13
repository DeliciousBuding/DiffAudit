import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.monitor_gsa_sequence import monitor_gsa_sequence, main


class MonitorGsaSequenceTests(unittest.TestCase):
    def test_monitor_gsa_sequence_training_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "checkpoints" / "target"
            target.mkdir(parents=True)
            (target / "checkpoint-1601").mkdir()
            (target / "train.stderr.log").write_text(
                "Epoch 12: ... loss=0.0415, step=384\n",
                encoding="utf-8",
            )

            payload = monitor_gsa_sequence(root)

            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["phase"], "training-target")
            self.assertEqual(payload["active_split"], "target")
            self.assertEqual(payload["splits"]["target"]["training_log"]["latest_epoch"], 12)
            self.assertEqual(payload["splits"]["target"]["training_log"]["latest_step"], 384)
            self.assertEqual(payload["splits"]["target"]["checkpoints"], ["checkpoint-1601"])

    def test_monitor_gsa_sequence_runtime_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime = root / "runs" / "runtime-001"
            runtime.mkdir(parents=True)
            (runtime / "summary.json").write_text("{}", encoding="utf-8")

            payload = monitor_gsa_sequence(root, runtime_workspace=runtime)

            self.assertEqual(payload["phase"], "runtime-ready")
            self.assertTrue(payload["runtime"]["summary_exists"])

    def test_cli_main(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "checkpoints" / "target"
            target.mkdir(parents=True)
            (target / "train.stderr.log").write_text(
                "Epoch 1: ... loss=0.1, step=32\n",
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                import sys

                original_argv = sys.argv
                try:
                    sys.argv = ["monitor_gsa_sequence.py", "--assets-root", str(root)]
                    main()
                finally:
                    sys.argv = original_argv

            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["phase"], "training-target")


if __name__ == "__main__":
    unittest.main()
