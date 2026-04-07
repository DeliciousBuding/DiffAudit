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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="validate_intake_index",
        description="Validate workspaces/intake/index.json and referenced manifests.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Path to intake index JSON (default: <project_root>/workspaces/intake/index.json).",
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    index_path = (args.index or (project_root / "workspaces" / "intake" / "index.json")).resolve()

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

                for key in ("id", "track", "method", "manifest", "admission", "compatibility"):
                    if key not in entry:
                        errors.append(f"entries[{i}] missing required field: {key}")

                manifest_rel = entry.get("manifest")
                if not isinstance(manifest_rel, str) or not manifest_rel:
                    errors.append(f"entries[{i}].manifest must be a non-empty string")
                    continue
                if not _is_rel_path(manifest_rel):
                    errors.append(f"entries[{i}].manifest must be a repo-relative path: {manifest_rel!r}")
                    continue

                manifest_path = (project_root / manifest_rel).resolve()
                if not manifest_path.exists():
                    errors.append(f"entries[{i}] manifest does not exist: {manifest_rel}")
                    continue

                manifest = _load_json(manifest_path)
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

                paths = entry.get("paths") or {}
                if isinstance(paths, dict) and "assets_root" in paths:
                    assets_root = paths["assets_root"]
                    if not _is_rel_path(assets_root):
                        errors.append(f"entries[{i}].paths.assets_root must be a repo-relative path: {assets_root!r}")
                    else:
                        abs_assets_root = (project_root / assets_root).resolve()
                        if not abs_assets_root.exists():
                            warnings.append(
                                f"entries[{i}] assets_root does not exist on this machine (ok if untracked): {assets_root}"
                            )

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

