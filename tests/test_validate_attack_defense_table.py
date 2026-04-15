import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def load_validate_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_attack_defense_table.py"
    spec = importlib.util.spec_from_file_location("validate_attack_defense_table_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ValidateAttackDefenseTableTests(unittest.TestCase):
    def test_project_attack_defense_table_is_currently_valid(self) -> None:
        module = load_validate_module()
        research_root = Path(__file__).resolve().parents[1]

        exit_code = module.main(
            [
                "--table",
                str(research_root / "workspaces" / "implementation" / "artifacts" / "unified-attack-defense-table.json"),
            ]
        )

        self.assertEqual(exit_code, 0)

    def test_project_table_promotes_latest_gsa_rerun_as_mainline(self) -> None:
        research_root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (research_root / "workspaces" / "implementation" / "artifacts" / "unified-attack-defense-table.json").read_text(
                encoding="utf-8"
            )
        )

        gsa_row = next(
            row
            for row in payload["rows"]
            if row.get("track") == "white-box" and row.get("attack") == "GSA 1k-3shadow" and row.get("defense") == "none"
        )

        self.assertEqual(
            gsa_row["source"],
            "workspaces/white-box/runs/gsa-runtime-mainline-20260409-cifar10-1k-3shadow-epoch300-rerun1/summary.json",
        )
        self.assertEqual(gsa_row["auc"], 0.998192)
        self.assertEqual(gsa_row["asr"], 0.9895)
        self.assertEqual(gsa_row["tpr_at_1pct_fpr"], 0.987)
        self.assertEqual(gsa_row["tpr_at_0_1pct_fpr"], 0.432)
        self.assertIn("not a final paper-level benchmark", gsa_row["boundary"])

    def test_project_table_rows_expose_boundary(self) -> None:
        research_root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (research_root / "workspaces" / "implementation" / "artifacts" / "unified-attack-defense-table.json").read_text(
                encoding="utf-8"
            )
        )

        for row in payload["rows"]:
            self.assertIn("boundary", row)
            self.assertTrue(row["boundary"])

    def test_validator_rejects_row_missing_required_field(self) -> None:
        module = load_validate_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.json"
            table_path.write_text(
                json.dumps(
                    {
                        "schema": "diffaudit.attack_defense_table.v1",
                        "updated_at": "2026-04-09T12:05:00+08:00",
                        "rows": [
                            {
                                "track": "black-box",
                                "attack": "recon DDIM public-100 step30",
                                "defense": "none",
                                "model": "Stable Diffusion v1.5 + DDIM",
                                "evidence_level": "runtime-mainline",
                                "boundary": "controlled / public-subset / risk-exists",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            exit_code = module.main(["--table", str(table_path)])

        self.assertEqual(exit_code, 2)

    def test_validator_rejects_absolute_source_paths(self) -> None:
        module = load_validate_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.json"
            table_path.write_text(
                json.dumps(
                    {
                        "schema": "diffaudit.attack_defense_table.v1",
                        "updated_at": "2026-04-09T12:05:00+08:00",
                        "rows": [
                            {
                                "track": "black-box",
                                "attack": "recon DDIM public-100 step30",
                                "defense": "none",
                                "model": "Stable Diffusion v1.5 + DDIM",
                                "evidence_level": "runtime-mainline",
                                "boundary": "controlled / public-subset / risk-exists",
                                "source": r"D:\Code\DiffAudit\Research\experiments\recon\summary.json",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            exit_code = module.main(["--table", str(table_path)])

        self.assertEqual(exit_code, 2)

    def test_validator_rejects_gray_box_row_without_adaptive_check(self) -> None:
        module = load_validate_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.json"
            table_path.write_text(
                json.dumps(
                    {
                        "schema": "diffaudit.attack_defense_table.v1",
                        "updated_at": "2026-04-09T12:05:00+08:00",
                        "rows": [
                            {
                                "track": "gray-box",
                                "attack": "PIA GPU512 baseline",
                                "defense": "none",
                                "model": "CIFAR-10 DDPM",
                                "evidence_level": "runtime-mainline",
                                "boundary": "workspace-verified + adaptive-reviewed + paper-alignment blocked by checkpoint/source provenance",
                                "source": "workspaces/gray-box/runs/pia/summary.json",
                                "quality": {"suite": "pia-runtime-surrogate-v1"},
                                "cost": {"device": "cuda:0"},
                                "provenance_status": "workspace-verified",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            exit_code = module.main(["--table", str(table_path)])

        self.assertEqual(exit_code, 2)

    def test_validator_rejects_gray_box_row_without_provenance_status(self) -> None:
        module = load_validate_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.json"
            table_path.write_text(
                json.dumps(
                    {
                        "schema": "diffaudit.attack_defense_table.v1",
                        "updated_at": "2026-04-09T12:05:00+08:00",
                        "rows": [
                            {
                                "track": "gray-box",
                                "attack": "PIA GPU512 baseline",
                                "defense": "none",
                                "model": "CIFAR-10 DDPM",
                                "evidence_level": "runtime-mainline",
                                "boundary": "workspace-verified + adaptive-reviewed + paper-alignment blocked by checkpoint/source provenance",
                                "source": "workspaces/gray-box/runs/pia/summary.json",
                                "quality": {"suite": "pia-runtime-surrogate-v1"},
                                "cost": {"device": "cuda:0"},
                                "adaptive_check": {
                                    "status": "completed",
                                    "query_repeats": 3,
                                    "aggregation": "mean",
                                    "metrics": {"auc": 0.84},
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            exit_code = module.main(["--table", str(table_path)])

        self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
