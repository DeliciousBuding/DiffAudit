import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.render_team_local_configs import render_all


TEAM_LOCAL = {
    "team": {"owner": "tester", "machine": "ci", "os": "windows"},
    "repo": {"research_root": "."},
    "black_box": {
        "recon": {
            "query_dataset_pt": "D:/datasets/recon/query.pt",
            "lora_checkpoint_dir": "D:/models/recon/lora",
        },
        "variation": {
            "query_image_root": "D:/datasets/variation/images",
            "endpoint": "http://127.0.0.1:9000/variation",
        },
        "clid": {
            "dataset_train_ref": "train-ref",
            "dataset_test_ref": "test-ref",
            "model_dir": "D:/models/clid",
        },
    },
    "gray_box": {
        "pia": {
            "dataset_parent": "D:/datasets/cifar10",
            "model_dir": "D:/models/pia",
            "member_split_root": "external/PIA/DDPM",
        },
        "secmi": {
            "dataset_root": "D:/datasets/secmi",
            "model_dir": "D:/models/secmi",
            "flagfile_path": "D:/models/secmi/flagfile.txt",
        },
    },
    "white_box": {
        "gsa": {"repo_root": "workspaces/white-box/external/GSA", "assets_root": "workspaces/white-box/assets/gsa"},
        "dpdm": {"repo_root": "external/DPDM"},
    },
}


class RenderTeamLocalConfigsTests(unittest.TestCase):
    def test_render_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            team_local_path = root / "team.local.yaml"
            output_dir = root / "rendered"
            team_local_path.write_text(yaml.safe_dump(TEAM_LOCAL, sort_keys=False), encoding="utf-8")

            paths = render_all(team_local_path, output_dir)

            self.assertEqual(len(paths), 5)
            variation = yaml.safe_load((output_dir / "variation.local.yaml").read_text(encoding="utf-8"))
            pia = yaml.safe_load((output_dir / "pia.local.yaml").read_text(encoding="utf-8"))
            recon = yaml.safe_load((output_dir / "recon.local.yaml").read_text(encoding="utf-8"))

            self.assertEqual(variation["assets"]["dataset_root"], "D:/datasets/variation/images")
            self.assertEqual(variation["attack"]["parameters"]["endpoint"], "http://127.0.0.1:9000/variation")
            self.assertEqual(pia["assets"]["model_dir"], "D:/models/pia")
            self.assertEqual(recon["assets"]["model_dir"], "D:/models/recon/lora")


if __name__ == "__main__":
    unittest.main()
