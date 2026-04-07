from __future__ import annotations

import hashlib
from pathlib import Path


def hash_file(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": path.name,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "mtime_ns": path.stat().st_mtime_ns,
    }


def hash_path(path: Path) -> dict:
    if path.is_file():
        file_info = hash_file(path)
        return {
            "kind": "file",
            "root": str(path),
            "algorithm": "sha256",
            "file_count": 1,
            "files": [file_info],
            "tree_sha256": file_info["sha256"],
        }

    files = []
    digests: list[str] = []
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = item.relative_to(path).as_posix()
        info = hash_file(item)
        info["path"] = rel
        files.append(info)
        digests.append(f"{rel}:{info['sha256']}")

    tree_sha = hashlib.sha256("\n".join(digests).encode("utf-8")).hexdigest()
    return {
        "kind": "dir",
        "root": str(path),
        "algorithm": "sha256",
        "file_count": len(files),
        "files": files,
        "tree_sha256": tree_sha,
    }
