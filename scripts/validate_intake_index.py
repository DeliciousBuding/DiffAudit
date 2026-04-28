from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_rel_path(p: str) -> bool:
    if not isinstance(p, str) or not p:
        return False
    # Intake paths are expected to be repo-relative (portable across machines).
    return not (p.startswith("/") or (len(p) >= 3 and p[1:3] == ":\\"))


def _extract_rel_paths(value: object) -> list[str]:
    """Best-effort: pull repo-relative path candidates out of manifest fields.

    The intake manifest uses a mix of string path values (single-root fields)
    and dicts of path values (e.g. datasets/checkpoint_roots).
    """

    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        paths: list[str] = []
        for v in value.values():
            if isinstance(v, str):
                paths.append(v)
        return paths
    return []


def _validate_optional_summary_path(
    errors: list[str],
    research_root: Path,
    entry_index: int,
    manifest: dict,
    field_name: str,
) -> None:
    value = manifest.get(field_name)
    if value is None:
        return
    if not isinstance(value, str) or not value:
        errors.append(f"entries[{entry_index}] manifest field {field_name!r} must be a non-empty string")
        return
    if not _is_rel_path(value):
        errors.append(f"entries[{entry_index}] manifest field {field_name!r} must be a repo-relative path: {value!r}")
        return
    if not (research_root / value).resolve().exists():
        errors.append(f"entries[{entry_index}] manifest field {field_name!r} does not exist: {value!r}")


def _validate_required_string(errors: list[str], prefix: str, payload: dict, field_name: str) -> str | None:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        errors.append(f"{prefix}.{field_name} must be a non-empty string")
        return None
    return value


def _validate_phase_e_source_doc(
    errors: list[str],
    research_root: Path,
    prefix: str,
    payload: dict,
) -> None:
    source_doc = _validate_required_string(errors, prefix, payload, "source_doc")
    if source_doc is None:
        return
    if not _is_rel_path(source_doc):
        errors.append(f"{prefix}.source_doc must be a repo-relative path: {source_doc!r}")
        return
    if not (research_root / source_doc).resolve().exists():
        errors.append(f"{prefix}.source_doc does not exist: {source_doc!r}")


def _validate_phase_e_candidate_file(errors: list[str], research_root: Path, candidate_path: Path) -> None:
    payload = _load_json(candidate_path)
    if payload.get("schema") != "diffaudit.phase_e_candidates.v2":
        errors.append(f"unsupported phase-e candidate schema: {payload.get('schema')!r}")
        return

    _validate_required_string(errors, "phase_e_candidates", payload, "updated_at")
    _validate_required_string(errors, "phase_e_candidates", payload, "status")
    _validate_required_string(errors, "phase_e_candidates", payload, "scope")

    document_layer = payload.get("document_layer_conditional")
    if not isinstance(document_layer, list) or not document_layer:
        errors.append("phase_e_candidates.document_layer_conditional must be a non-empty list")
    else:
        for i, item in enumerate(document_layer):
            prefix = f"phase_e_candidates.document_layer_conditional[{i}]"
            if not isinstance(item, dict):
                errors.append(f"{prefix} must be an object")
                continue
            for field_name in ("id", "track", "record_class", "current_verdict", "current_shape", "current_boundary"):
                _validate_required_string(errors, prefix, item, field_name)
            _validate_phase_e_source_doc(errors, research_root, prefix, item)
            for forbidden in ("contract_key", "manifest", "compatibility", "admission"):
                if forbidden in item:
                    errors.append(f"{prefix} must not define {forbidden!r}")

    status = payload.get("status")
    execution_layer = payload.get("intake_review_priority_order")
    allow_empty_execution_layer = status == "document-conditional-only"
    if not isinstance(execution_layer, list):
        errors.append("phase_e_candidates.intake_review_priority_order must be a list")
    elif not execution_layer and not allow_empty_execution_layer:
        errors.append("phase_e_candidates.intake_review_priority_order must be a non-empty list")
    else:
        expected_order = 1
        for i, item in enumerate(execution_layer):
            prefix = f"phase_e_candidates.intake_review_priority_order[{i}]"
            if not isinstance(item, dict):
                errors.append(f"{prefix} must be an object")
                continue
            order = item.get("order")
            if not isinstance(order, int):
                errors.append(f"{prefix}.order must be an integer")
            elif order != expected_order:
                errors.append(f"{prefix}.order must equal {expected_order}")
            expected_order += 1
            for field_name in ("id", "track", "record_class", "current_verdict", "current_shape", "current_boundary"):
                _validate_required_string(errors, prefix, item, field_name)
            _validate_phase_e_source_doc(errors, research_root, prefix, item)
            for forbidden in ("contract_key", "manifest", "compatibility", "admission"):
                if forbidden in item:
                    errors.append(f"{prefix} must not define {forbidden!r}")


def _derive_research_root(index_path: Path, fallback_root: Path) -> Path:
    if index_path.parent.name == "intake" and index_path.parent.parent.name == "workspaces":
        return index_path.parent.parent.parent
    return index_path.parent if index_path.parent != index_path else fallback_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="validate_intake_index",
        description="Validate workspaces/intake/index.json and referenced manifests.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Path to intake index JSON (default: <research_root>/workspaces/intake/index.json).",
    )
    args = parser.parse_args(argv)

    fallback_root = Path(__file__).resolve().parents[1]
    index_path = (args.index or (fallback_root / "workspaces" / "intake" / "index.json")).resolve()
    research_root = _derive_research_root(index_path, fallback_root)

    errors: list[str] = []
    warnings: list[str] = []

    if not index_path.exists():
        errors.append(f"index does not exist: {index_path}")
    else:
        index = _load_json(index_path)
        if index.get("schema") != "diffaudit.intake.index.v1":
            errors.append(f"unsupported schema: {index.get('schema')!r}")

        entries = index.get("entries")
        if not isinstance(entries, list) or not entries:
            errors.append("entries must be a non-empty list")
        else:
            for i, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    errors.append(f"entries[{i}] must be an object")
                    continue

                for key in ("id", "contract_key", "track", "method", "manifest", "admission", "compatibility"):
                    if key not in entry:
                        errors.append(f"entries[{i}] missing required field: {key}")

                contract_key = entry.get("contract_key")
                if not isinstance(contract_key, str) or not contract_key:
                    errors.append(f"entries[{i}].contract_key must be a non-empty string")

                manifest_rel = entry.get("manifest")
                if not isinstance(manifest_rel, str) or not manifest_rel:
                    errors.append(f"entries[{i}].manifest must be a non-empty string")
                    continue
                if not _is_rel_path(manifest_rel):
                    errors.append(f"entries[{i}].manifest must be a repo-relative path: {manifest_rel!r}")
                    continue

                manifest_path = (research_root / manifest_rel).resolve()
                if not manifest_path.exists():
                    errors.append(f"entries[{i}] manifest does not exist: {manifest_rel}")
                    continue

                manifest = _load_json(manifest_path)
                if manifest.get("schema") != "diffaudit.intake.manifest.v1":
                    errors.append(
                        f"entries[{i}] manifest unsupported schema: {manifest.get('schema')!r}"
                    )
                if manifest.get("contract_key") != entry.get("contract_key"):
                    errors.append(
                        f"entries[{i}] contract_key mismatch: entry={entry.get('contract_key')!r} "
                        f"manifest={manifest.get('contract_key')!r}"
                    )
                if manifest.get("track") != entry.get("track"):
                    errors.append(
                        f"entries[{i}] track mismatch: entry={entry.get('track')!r} manifest={manifest.get('track')!r}"
                    )
                if manifest.get("method") != entry.get("method"):
                    errors.append(
                        f"entries[{i}] method mismatch: entry={entry.get('method')!r} manifest={manifest.get('method')!r}"
                    )

                for key in ("contract_stage", "asset_grade", "provenance_status", "evidence_level"):
                    if key not in manifest:
                        errors.append(f"entries[{i}] manifest missing required field: {key}")

                _validate_optional_summary_path(errors, research_root, i, manifest, "canonical_summary")
                _validate_optional_summary_path(errors, research_root, i, manifest, "defense_summary")

                admission = entry.get("admission") or {}
                if isinstance(admission, dict):
                    for k in ("evidence_level", "provenance_status"):
                        if k in admission and k in manifest and admission[k] != manifest[k]:
                            errors.append(
                                f"entries[{i}] admission.{k} mismatch: admission={admission[k]!r} manifest={manifest[k]!r}"
                            )

                compat = entry.get("compatibility") or {}
                if isinstance(compat, dict):
                    commands = compat.get("commands") or []
                    if not isinstance(commands, list) or not commands:
                        errors.append(f"entries[{i}] compatibility.commands must be a non-empty list")
                    else:
                        for j, cmd in enumerate(commands):
                            if not isinstance(cmd, dict):
                                errors.append(f"entries[{i}].compatibility.commands[{j}] must be an object")
                                continue
                            required = cmd.get("required_manifest_fields") or []
                            if not isinstance(required, list) or not required:
                                errors.append(
                                    f"entries[{i}].compatibility.commands[{j}].required_manifest_fields "
                                    "must be a non-empty list"
                                )
                                continue
                            missing = [k for k in required if k not in manifest]
                            if missing:
                                errors.append(
                                    f"entries[{i}] manifest missing fields required by "
                                    f"compatibility.commands[{j}]: {missing}"
                                )
                            for field in required:
                                for path_value in _extract_rel_paths(manifest.get(field)):
                                    if not _is_rel_path(path_value):
                                        errors.append(
                                            f"entries[{i}] manifest field {field!r} must contain repo-relative paths: "
                                            f"{path_value!r}"
                                        )

                paths = entry.get("paths") or {}
                if isinstance(paths, dict) and "assets_root" in paths:
                    assets_root = paths["assets_root"]
                    if not _is_rel_path(assets_root):
                        errors.append(f"entries[{i}].paths.assets_root must be a repo-relative path: {assets_root!r}")
                    else:
                        abs_assets_root = (research_root / assets_root).resolve()
                        if not abs_assets_root.exists():
                            warnings.append(
                                f"entries[{i}] assets_root does not exist on this machine (ok if untracked): {assets_root}"
                            )

        candidate_path = index_path.parent / "phase-e-candidates.json"
        if candidate_path.exists():
            _validate_phase_e_candidate_file(errors, research_root, candidate_path)

    if warnings:
        for w in warnings:
            print(f"WARN {w}")
    if errors:
        for e in errors:
            print(f"ERROR {e}")
        return 2

    print(f"OK {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
