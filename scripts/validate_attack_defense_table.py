from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_rel_path(value: str) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return not (value.startswith("/") or (len(value) >= 3 and value[1:3] == ":\\"))  # repo-relative only


def _require_fields(errors: list[str], prefix: str, payload: dict, required: tuple[str, ...]) -> None:
    for key in required:
        if key not in payload:
            errors.append(f"{prefix} missing required field: {key}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="validate_attack_defense_table",
        description="Validate the admitted unified attack-defense table contract.",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=None,
        help="Path to unified attack-defense table JSON (default: <research_root>/workspaces/implementation/artifacts/unified-attack-defense-table.json).",
    )
    args = parser.parse_args(argv)

    research_root = Path(__file__).resolve().parents[1]
    table_path = (
        args.table
        or research_root / "workspaces" / "implementation" / "artifacts" / "unified-attack-defense-table.json"
    ).resolve()

    errors: list[str] = []

    if not table_path.exists():
        errors.append(f"table does not exist: {table_path}")
    else:
        payload = _load_json(table_path)
        if payload.get("schema") != "diffaudit.attack_defense_table.v1":
            errors.append(f"unsupported schema: {payload.get('schema')!r}")
        _require_fields(errors, "table", payload, ("schema", "updated_at", "rows"))

        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            errors.append("rows must be a non-empty list")
        else:
            for idx, row in enumerate(rows):
                if not isinstance(row, dict):
                    errors.append(f"rows[{idx}] must be an object")
                    continue

                _require_fields(
                    errors,
                    f"rows[{idx}]",
                    row,
                    ("track", "attack", "defense", "model", "evidence_level", "source", "boundary"),
                )

                source = row.get("source")
                if isinstance(source, str) and source and not _is_rel_path(source):
                    errors.append(f"rows[{idx}].source must be a repo-relative path: {source!r}")

                if row.get("track") != "gray-box":
                    continue

                _require_fields(
                    errors,
                    f"rows[{idx}]",
                    row,
                    ("quality", "cost", "adaptive_check", "provenance_status"),
                )

                adaptive_check = row.get("adaptive_check")
                if not isinstance(adaptive_check, dict):
                    continue
                if adaptive_check.get("status") != "completed":
                    continue

                _require_fields(
                    errors,
                    f"rows[{idx}].adaptive_check",
                    adaptive_check,
                    ("query_repeats", "aggregation", "metrics"),
                )

    if errors:
        for error in errors:
            print(f"ERROR {error}")
        return 2

    print(f"OK {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
