"""Initialize a minimal real-asset query-image layout for variation attacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


README_TEXT = """# Variation Query Image Set

This directory is a minimal scaffold for a real variation-query asset set.

Expected layout:

- member/
- nonmember/

Put real query images into each split before claiming the variation line is
asset-ready.
"""


def init_variation_query_set(root: str | Path) -> dict[str, object]:
    root_path = Path(root)
    member_dir = root_path / "member"
    nonmember_dir = root_path / "nonmember"
    member_dir.mkdir(parents=True, exist_ok=True)
    nonmember_dir.mkdir(parents=True, exist_ok=True)

    readme_path = root_path / "README.md"
    readme_path.write_text(README_TEXT, encoding="utf-8")

    member_hint = member_dir / "PLACE_MEMBER_IMAGES_HERE.txt"
    nonmember_hint = nonmember_dir / "PLACE_NONMEMBER_IMAGES_HERE.txt"
    if not member_hint.exists():
        member_hint.write_text(
            "Put real member query images here.\n",
            encoding="utf-8",
        )
    if not nonmember_hint.exists():
        nonmember_hint.write_text(
            "Put real non-member query images here.\n",
            encoding="utf-8",
        )

    return {
        "status": "ready",
        "root": str(root_path),
        "paths": {
            "root": str(root_path),
            "member_dir": str(member_dir),
            "nonmember_dir": str(nonmember_dir),
            "readme": str(readme_path),
        },
        "checks": {
            "member_dir": member_dir.exists(),
            "nonmember_dir": nonmember_dir.exists(),
            "readme": readme_path.exists(),
        },
        "notes": [
            "This scaffold only prepares the directory shape for a real variation query set.",
            "The variation line remains blocked until real images and a real endpoint are both available.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="target directory for the query-image scaffold")
    args = parser.parse_args()

    payload = init_variation_query_set(args.root)
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
