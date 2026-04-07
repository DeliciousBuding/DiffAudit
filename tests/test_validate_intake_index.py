import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def load_validate_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_intake_index.py"
    spec = importlib.util.spec_from_file_location("validate_intake_index_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ValidateIntakeIndexTests(unittest.TestCase):
    def test_project_intake_index_declares_contract_keys(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        payload = json.loads((project_root / "workspaces" / "intake" / "index.json").read_text(encoding="utf-8"))

        contract_keys = {entry.get("contract_key") for entry in payload["entries"]}

        self.assertIn("gray-box/pia/cifar10-ddpm", contract_keys)
        self.assertIn("white-box/gsa/ddpm-cifar10", contract_keys)

    def test_validator_rejects_entry_without_contract_key(self) -> None:
        module = load_validate_module()
        project_root = Path(__file__).resolve().parents[1]
        manifest_rel = "workspaces/gray-box/assets/pia/manifest.json"

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "schema": "diffaudit.intake.index.v1",
                        "updated_at": "2026-04-07",
                        "entries": [
                            {
                                "id": "gray-box/pia/cifar10-ddpm",
                                "track": "gray-box",
                                "method": "pia",
                                "paths": {"assets_root": "workspaces/gray-box/assets/pia"},
                                "manifest": manifest_rel,
                                "admission": {
                                    "status": "admitted",
                                    "level": "system-intake-ready",
                                    "evidence_level": "runtime-mainline",
                                    "provenance_status": "source-retained-unverified",
                                },
                                "compatibility": {
                                    "surface": "diffaudit-cli",
                                    "commands": [
                                        {
                                            "name": "run-pia-runtime-mainline",
                                            "required_manifest_fields": [
                                                "checkpoint_root",
                                                "dataset_root",
                                                "member_split_root",
                                                "member_split_file",
                                            ],
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            exit_code = module.main(["--index", str(index_path)])

        self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
