from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hashing import HashedFile


@dataclass
class TreeHash:
    root: str
    algorithm: str
    file_count: int
    tree_sha256: str
    files: list[HashedFile]

    def to_json(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "algorithm": self.algorithm,
            "file_count": self.file_count,
            "tree_sha256": self.tree_sha256,
            "files": [f.to_json() for f in self.files],
        }


def config_root_label(config_path: Path) -> str:
    p = config_path.resolve()
    return str(p)

