from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .gitinfo import detect_git
from .hashing import hash_tree, sha256_bytes, sha256_file
from .model import TreeHash
from .util import CheckResult, host_info, json_dumps_canonical, now_iso_local, norm_abs


def _hash_config(config_path: Path) -> tuple[dict[str, Any], str]:
    config_path = config_path.resolve()
    if config_path.is_file():
        content_sha = sha256_file(config_path)
        payload = {
            "kind": "file",
            "path": str(config_path),
            "algorithm": "sha256",
            "sha256": content_sha,
            "size": int(config_path.stat().st_size),
        }
        return payload, content_sha

    files, tree_sha = hash_tree(config_path)
    tree = TreeHash(
        root=str(config_path),
        algorithm="sha256",
        file_count=len(files),
        tree_sha256=tree_sha,
        files=files,
    )
    payload = {"kind": "dir", **tree.to_json()}
    return payload, tree_sha


def run_gate(
    *,
    assets_root: Path,
    repo_root: Path,
    config_path: Path,
    strict: bool,
) -> tuple[CheckResult, dict[str, Any], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []

    assets_root = assets_root.resolve()
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()

    if not assets_root.exists() or not assets_root.is_dir():
        errors.append(f"assets_root is not an existing directory: {assets_root}")

    if not repo_root.exists() or not repo_root.is_dir():
        errors.append(f"repo_root is not an existing directory: {repo_root}")

    if not config_path.exists():
        errors.append(f"config does not exist: {config_path}")
    else:
        if not (config_path.is_file() or config_path.is_dir()):
            errors.append(f"config must be a file or directory: {config_path}")

    git = detect_git(repo_root)
    if strict:
        if not git.available:
            errors.append("git is not available on PATH but --strict was set")
        elif not git.is_repo:
            errors.append("repo_root is not a git work tree but --strict was set")
        elif git.commit is None:
            errors.append("could not read git commit but --strict was set")
        elif git.dirty:
            errors.append("repo_root has uncommitted changes but --strict was set")
    else:
        if git.available and git.is_repo and git.dirty:
            warnings.append("repo_root has uncommitted changes (git status not clean)")

    assets_files: list[Any] = []
    assets_tree_sha: str | None = None
    if assets_root.exists() and assets_root.is_dir():
        files, tree_sha = hash_tree(assets_root)
        assets_files = [f.to_json() for f in files]
        assets_tree_sha = tree_sha

    config_payload: dict[str, Any] | None = None
    config_hash: str | None = None
    if config_path.exists() and (config_path.is_file() or config_path.is_dir()):
        config_payload, config_hash = _hash_config(config_path)

    ok = len(errors) == 0
    if strict and not ok:
        warnings = []  # keep output focused in strict mode

    check = CheckResult(ok=ok, errors=errors, warnings=warnings)

    manifest: dict[str, Any] = {
        "schema": "gsa_next_run.manifest.v1",
        "created_at": now_iso_local(),
        "paths": {
            "assets_root": norm_abs(assets_root),
            "repo_root": norm_abs(repo_root),
            "config": norm_abs(config_path),
        },
        "inputs": {
            "assets": {
                "algorithm": "sha256",
                "root": str(assets_root),
                "file_count": len(assets_files),
                "tree_sha256": assets_tree_sha,
                "files": assets_files,
            },
            "config": config_payload,
            "config_sha256": config_hash,
        },
        "git": git.to_json(),
        "validation": check.to_json(),
    }

    manifest_sha256 = sha256_bytes(json_dumps_canonical(manifest))

    provenance: dict[str, Any] = {
        "schema": "gsa_next_run.provenance.v1",
        "created_at": now_iso_local(),
        "manifest_sha256": manifest_sha256,
        "host": host_info(),
        "validation": check.to_json(),
    }

    return check, manifest, provenance


def write_outputs(*, out_dir: Path, manifest: dict[str, Any], provenance: dict[str, Any]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    provenance_path = out_dir / "provenance.json"

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path, provenance_path

