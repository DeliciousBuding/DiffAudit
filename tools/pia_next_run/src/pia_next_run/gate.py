from __future__ import annotations

import json
import os
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .gitinfo import read_git_info
from .hashing import hash_path


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]


def run_gate(config_path: Path, member_split_root: Path, repo_root: Path, strict: bool = False):
    errors: list[str] = []
    warnings: list[str] = []

    if not config_path.exists():
        errors.append(f"config does not exist: {config_path}")
    if not member_split_root.exists():
        errors.append(f"member_split_root does not exist: {member_split_root}")
    if not repo_root.exists():
        errors.append(f"repo_root does not exist: {repo_root}")

    git = read_git_info(repo_root) if repo_root.exists() else {
        "available": False,
        "is_repo": False,
        "commit": None,
        "describe": None,
        "dirty": None,
        "error": "repo_root missing",
    }
    if git.get("dirty"):
        warnings.append("repo_root has uncommitted changes (git status not clean)")
    if strict and not git.get("is_repo"):
        errors.append("repo_root is not a git work tree")
    if strict and git.get("dirty"):
        errors.append("repo_root is dirty")

    split_file = member_split_root / "CIFAR10_train_ratio0.5.npz"
    if member_split_root.exists() and not split_file.exists():
        warnings.append("expected member split file missing: CIFAR10_train_ratio0.5.npz")

    ok = not errors
    manifest = {
        "schema": "pia_next_run.manifest.v1",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "paths": {
            "config": str(config_path),
            "member_split_root": str(member_split_root),
            "repo_root": str(repo_root),
        },
        "git": git,
        "inputs": {
            "config": hash_path(config_path) if config_path.exists() else None,
            "member_split_root": hash_path(member_split_root) if member_split_root.exists() else None,
        },
        "validation": {
            "ok": ok,
            "errors": errors,
            "warnings": warnings,
        },
    }

    manifest_json = json.dumps(manifest, indent=2, sort_keys=True)
    provenance = {
        "schema": "pia_next_run.provenance.v1",
        "created_at": manifest["created_at"],
        "manifest_sha256": __import__("hashlib").sha256(manifest_json.encode("utf-8")).hexdigest(),
        "host": {
            "cwd": os.getcwd(),
            "hostname": socket.gethostname(),
            "platform": sys.platform,
            "python": sys.version,
            "pid": os.getpid(),
        },
        "validation": manifest["validation"],
    }
    return ValidationResult(ok=ok, errors=errors, warnings=warnings), manifest, provenance


def write_outputs(out_dir: Path, manifest: dict, provenance: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    provenance_path = out_dir / "provenance.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path, provenance_path
