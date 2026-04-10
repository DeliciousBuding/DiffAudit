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

    def test_project_gsa_intake_points_to_1k_3shadow_mainline(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        payload = json.loads((project_root / "workspaces" / "intake" / "index.json").read_text(encoding="utf-8"))

        gsa_entry = next(entry for entry in payload["entries"] if entry.get("contract_key") == "white-box/gsa/ddpm-cifar10")
        self.assertEqual(
            gsa_entry["paths"]["assets_root"],
            "workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1",
        )
        self.assertEqual(
            gsa_entry["manifest"],
            "workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/manifests/cifar10-ddpm-1k-3shadow-epoch300-rerun1.json",
        )

        manifest_path = project_root / Path(gsa_entry["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["contract_key"], "white-box/gsa/ddpm-cifar10")
        self.assertEqual(manifest["canonical_summary"], "workspaces/white-box/runs/gsa-runtime-mainline-20260409-cifar10-1k-3shadow-epoch300-rerun1/summary.json")
        self.assertEqual(manifest["evidence_level"], "runtime-mainline")

    def test_project_phase_e_candidate_registry_preserves_candidate_boundary(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (project_root / "workspaces" / "intake" / "phase-e-candidates.json").read_text(encoding="utf-8")
        )

        self.assertEqual(payload["schema"], "diffaudit.phase_e_candidates.v1")
        self.assertEqual(payload["status"], "intake-only")

        document_layer = payload["document_layer_conditional"]
        self.assertEqual(len(document_layer), 1)
        self.assertEqual(document_layer[0]["id"], "gray-box/pia/paper-aligned-confirmation")
        self.assertEqual(document_layer[0]["execution_layer_status"], "no-go")
        self.assertNotIn("contract_key", document_layer[0])
        self.assertNotIn("manifest", document_layer[0])

        execution_layer = payload["execution_layer_default_order"]
        self.assertEqual([item["order"] for item in execution_layer], [1, 2, 3, 4])
        self.assertEqual(execution_layer[0]["current_shape"], "adapter-complete zero-GPU hold")
        self.assertEqual(execution_layer[0]["queue_state"], "not-requestable")
        self.assertEqual(execution_layer[1]["current_shape"], "comparability / intake hardening only")
        self.assertEqual(execution_layer[2]["current_shape"], "blocked baseline")
        self.assertEqual(execution_layer[3]["current_shape"], "protocol-and-asset decomposition intake only")
        for item in execution_layer:
            self.assertNotIn("contract_key", item)
            self.assertNotIn("manifest", item)

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

    def test_validator_rejects_manifest_with_absolute_canonical_summary(self) -> None:
        module = load_validate_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            manifest_rel = "workspaces/white-box/assets/gsa/manifests/cifar10-ddpm-mainline.json"
            manifest_path = temp_root / manifest_rel
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": "diffaudit.intake.manifest.v1",
                        "contract_key": "white-box/gsa/ddpm-cifar10",
                        "track": "white-box",
                        "method": "gsa",
                        "contract_stage": "target",
                        "asset_grade": "asset-ready",
                        "provenance_status": "workspace-verified",
                        "evidence_level": "runtime-mainline",
                        "datasets": {
                            "target_member": "workspaces/white-box/assets/gsa/datasets/target-member",
                        },
                        "checkpoint_roots": {
                            "target": "workspaces/white-box/assets/gsa/checkpoints/target",
                        },
                        "checkpoint_format": "accelerate-checkpoint-dir",
                        "canonical_summary": r"D:\Code\DiffAudit\Project\workspaces\white-box\runs\gsa-runtime-mainline-20260408-cifar10-1k-3shadow\summary.json",
                    }
                ),
                encoding="utf-8",
            )
            index_path = temp_root / "workspaces" / "intake" / "index.json"
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_text(
                json.dumps(
                    {
                        "schema": "diffaudit.intake.index.v1",
                        "updated_at": "2026-04-09",
                        "entries": [
                            {
                                "id": "white-box/gsa/cifar10-ddpm",
                                "contract_key": "white-box/gsa/ddpm-cifar10",
                                "track": "white-box",
                                "method": "gsa",
                                "paths": {
                                    "assets_root": "workspaces/white-box/assets/gsa",
                                },
                                "manifest": manifest_rel,
                                "admission": {
                                    "status": "admitted",
                                    "level": "system-intake-ready",
                                    "evidence_level": "runtime-mainline",
                                    "provenance_status": "workspace-verified",
                                },
                                "compatibility": {
                                    "surface": "diffaudit-cli",
                                    "commands": [
                                        {
                                            "name": "run-gsa-runtime-mainline",
                                            "required_manifest_fields": ["datasets", "checkpoint_roots", "checkpoint_format"],
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

    def test_validator_rejects_phase_e_candidate_with_contract_key(self) -> None:
        module = load_validate_module()
        project_root = Path(__file__).resolve().parents[1]
        manifest_rel = "workspaces/gray-box/assets/pia/manifest.json"

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            source_doc_rel = "workspaces/intake/phase-e-note.md"
            source_doc_path = temp_root / source_doc_rel
            source_doc_path.parent.mkdir(parents=True, exist_ok=True)
            source_doc_path.write_text("# note\n", encoding="utf-8")

            index_path = temp_root / "workspaces" / "intake" / "index.json"
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_text(
                json.dumps(
                    {
                        "schema": "diffaudit.intake.index.v1",
                        "updated_at": "2026-04-10",
                        "entries": [
                            {
                                "id": "gray-box/pia/cifar10-ddpm",
                                "contract_key": "gray-box/pia/cifar10-ddpm",
                                "track": "gray-box",
                                "method": "pia",
                                "paths": {"assets_root": "workspaces/gray-box/assets/pia"},
                                "manifest": manifest_rel,
                                "admission": {
                                    "status": "admitted",
                                    "level": "system-intake-ready",
                                    "evidence_level": "runtime-mainline",
                                    "provenance_status": "workspace-verified",
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
            candidate_path = temp_root / "workspaces" / "intake" / "phase-e-candidates.json"
            candidate_path.write_text(
                json.dumps(
                    {
                        "schema": "diffaudit.phase_e_candidates.v1",
                        "updated_at": "2026-04-10",
                        "status": "intake-only",
                        "scope": "candidate ordering only",
                        "document_layer_conditional": [
                            {
                                "id": "gray-box/pia/paper-aligned-confirmation",
                                "track": "gray-box",
                                "record_class": "phase-e-document-conditional",
                                "current_verdict": "blocked",
                                "current_shape": "document-layer conditional rank 1",
                                "execution_layer_status": "no-go",
                                "source_doc": source_doc_rel,
                            }
                        ],
                        "execution_layer_default_order": [
                            {
                                "order": 1,
                                "id": "white-box/finding-nemo",
                                "track": "white-box",
                                "record_class": "phase-e-candidate",
                                "current_verdict": "not-yet",
                                "current_shape": "adapter-complete zero-GPU hold",
                                "execution_release": "none",
                                "gpu_release": "none",
                                "source_doc": source_doc_rel,
                                "contract_key": "white-box/gsa/ddpm-cifar10",
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
