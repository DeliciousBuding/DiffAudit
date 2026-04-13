import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout

from scripts.init_variation_query_set import init_variation_query_set, main


class InitVariationQuerySetTests(unittest.TestCase):
    def test_init_variation_query_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "variation-query-set"
            payload = init_variation_query_set(root)

            self.assertEqual(payload["status"], "ready")
            self.assertTrue((root / "member").exists())
            self.assertTrue((root / "nonmember").exists())
            self.assertTrue((root / "README.md").exists())
            self.assertTrue((root / "member" / "PLACE_MEMBER_IMAGES_HERE.txt").exists())
            self.assertTrue((root / "nonmember" / "PLACE_NONMEMBER_IMAGES_HERE.txt").exists())

    def test_cli_main(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "variation-query-set"
            stdout = StringIO()
            with redirect_stdout(stdout):
                import sys

                original_argv = sys.argv
                try:
                    sys.argv = ["init_variation_query_set.py", "--root", str(root)]
                    main()
                finally:
                    sys.argv = original_argv

            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "ready")
            self.assertTrue((root / "README.md").exists())


if __name__ == "__main__":
    unittest.main()
