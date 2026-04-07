from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass(frozen=True)
class HashedFile:
    relpath: str
    sha256: str
    size: int
    mtime_ns: int

    def to_json(self) -> dict:
        return {
            "path": self.relpath,
            "sha256": self.sha256,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
        }


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def hash_tree(root: Path) -> tuple[list[HashedFile], str]:
    root = root.resolve()
    files: list[HashedFile] = []
    for p in iter_files(root):
        st = p.stat()
        rel = str(p.relative_to(root)).replace("\\", "/")
        files.append(
            HashedFile(
                relpath=rel,
                sha256=sha256_file(p),
                size=int(st.st_size),
                mtime_ns=int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
            )
        )
    files.sort(key=lambda x: x.relpath)

    # Deterministic tree digest based on file paths + file sha256.
    h = hashlib.sha256()
    for f in files:
        h.update(f.relpath.encode("utf-8"))
        h.update(b"\0")
        h.update(f.sha256.encode("ascii"))
        h.update(b"\n")
    return files, h.hexdigest()

