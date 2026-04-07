from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _norm_rel(p: str) -> str:
    return p.replace("\\", "/").lstrip("./")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="validate_local_api_registry_alignment",
        description="Validate Project intake index alignment with Services/Local-API SQLite registry seed.",
    )
    parser.add_argument(
        "--intake-index",
        type=Path,
        default=None,
        help="Path to Project workspaces/intake/index.json.",
    )
    parser.add_argument(
        "--registry-seed",
        type=Path,
        default=None,
        help="Path to Services/Local-API internal/api/registry_seed.json.",
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent

    intake_index_path = (args.intake_index or (project_root / "workspaces" / "intake" / "index.json")).resolve()
    registry_seed_path = (
        args.registry_seed
        or (workspace_root / "Services" / "Local-API" / "internal" / "api" / "registry_seed.json")
    ).resolve()

    errors: list[str] = []

    if not intake_index_path.exists():
        errors.append(f"intake index does not exist: {intake_index_path}")
    if not registry_seed_path.exists():
        errors.append(f"registry seed does not exist: {registry_seed_path}")
    if errors:
        for e in errors:
            print(f"ERROR {e}")
        return 2

    intake_index = _load_json(intake_index_path)
    if not isinstance(intake_index, dict) or "entries" not in intake_index:
        print(f"ERROR invalid intake index payload: {intake_index_path}")
        return 2

    seed = _load_json(registry_seed_path)
    if not isinstance(seed, list) or not seed:
        print(f"ERROR invalid registry seed payload: {registry_seed_path}")
        return 2

    contracts_by_key: dict[str, dict] = {}
    for item in seed:
        if not isinstance(item, dict):
            continue
        key = item.get("contract_key")
        if isinstance(key, str) and key:
            contracts_by_key[key] = item

    entries = intake_index.get("entries")
    if not isinstance(entries, list) or not entries:
        print(f"ERROR intake index entries must be a non-empty list: {intake_index_path}")
        return 2

    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"entries[{i}] must be an object")
            continue

        contract_key = entry.get("contract_key")
        if not isinstance(contract_key, str) or not contract_key:
            errors.append(f"entries[{i}].contract_key must be a non-empty string")
            continue

        contract = contracts_by_key.get(contract_key)
        if contract is None:
            errors.append(f"entries[{i}] contract_key not found in Local-API registry seed: {contract_key!r}")
            continue

        entry_track = entry.get("track")
        entry_method = entry.get("method")
        contract_track = contract.get("track")
        contract_method = contract.get("attack_family")

        if entry_track != contract_track:
            errors.append(
                f"entries[{i}] track mismatch vs registry seed: entry={entry_track!r} seed={contract_track!r}"
            )
        if entry_method != contract_method:
            errors.append(
                f"entries[{i}] method mismatch vs registry seed: entry={entry_method!r} seed={contract_method!r}"
            )

        paths = entry.get("paths") or {}
        assets_root = None
        if isinstance(paths, dict):
            assets_root = paths.get("assets_root")
        if isinstance(assets_root, str) and assets_root:
            assets_root_norm = _norm_rel(assets_root)
            promoted = contract.get("promoted_asset_roots") or []
            if not isinstance(promoted, list):
                errors.append(f"entries[{i}] registry seed promoted_asset_roots must be a list for {contract_key!r}")
                continue

            for j, promoted_root in enumerate(promoted):
                if not isinstance(promoted_root, str) or not promoted_root:
                    errors.append(
                        f"entries[{i}] registry seed promoted_asset_roots[{j}] must be a non-empty string for {contract_key!r}"
                    )
                    continue
                promoted_norm = _norm_rel(promoted_root)
                if promoted_norm.startswith("Project/"):
                    promoted_norm = promoted_norm[len("Project/") :]
                if promoted_norm == assets_root_norm:
                    continue
                if promoted_norm.startswith(assets_root_norm.rstrip("/") + "/"):
                    continue
                errors.append(
                    f"entries[{i}] promoted_asset_root not under intake paths.assets_root: "
                    f"assets_root={assets_root_norm!r} promoted={promoted_norm!r} contract_key={contract_key!r}"
                )

        required_now = contract.get("required_inputs_now") or []
        if isinstance(required_now, list) and "assets_root" in required_now:
            if not (isinstance(assets_root, str) and assets_root):
                errors.append(
                    f"entries[{i}] contract requires assets_root but intake entry missing paths.assets_root: {contract_key!r}"
                )

    if errors:
        for e in errors:
            print(f"ERROR {e}")
        return 2

    print(f"OK intake={intake_index_path} seed={registry_seed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

