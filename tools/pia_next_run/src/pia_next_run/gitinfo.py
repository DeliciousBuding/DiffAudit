from __future__ import annotations

import subprocess
from pathlib import Path


def read_git_info(repo_root: Path) -> dict:
    result = {
        "available": False,
        "is_repo": False,
        "commit": None,
        "describe": None,
        "dirty": None,
        "error": None,
    }
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        result["available"] = True
        result["is_repo"] = True
        result["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        result["describe"] = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        result["dirty"] = bool(status)
    except Exception as exc:  # pragma: no cover - defensive
        result["error"] = str(exc)
    return result
