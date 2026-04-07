from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _run_git(repo_root: Path, args: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return p.returncode, p.stdout.strip(), p.stderr.strip()


@dataclass
class GitInfo:
    available: bool
    is_repo: bool
    commit: str | None
    describe: str | None
    dirty: bool | None
    error: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "is_repo": self.is_repo,
            "commit": self.commit,
            "describe": self.describe,
            "dirty": self.dirty,
            "error": self.error,
        }


def detect_git(repo_root: Path) -> GitInfo:
    repo_root = repo_root.resolve()

    try:
        code, out, err = _run_git(repo_root, ["rev-parse", "--is-inside-work-tree"])
    except FileNotFoundError as e:
        return GitInfo(
            available=False,
            is_repo=False,
            commit=None,
            describe=None,
            dirty=None,
            error=str(e),
        )

    if code != 0 or out.lower() != "true":
        return GitInfo(
            available=True,
            is_repo=False,
            commit=None,
            describe=None,
            dirty=None,
            error=err or "repo_root is not a git work tree",
        )

    code, commit, err2 = _run_git(repo_root, ["rev-parse", "HEAD"])
    commit_val = commit if code == 0 else None
    err_commit = err2 if code != 0 else None

    code, desc, _ = _run_git(repo_root, ["describe", "--always", "--dirty"])
    desc_val = desc if code == 0 else None

    code, status, _ = _run_git(repo_root, ["status", "--porcelain"])
    dirty_val = None
    if code == 0:
        dirty_val = bool(status.strip())

    return GitInfo(
        available=True,
        is_repo=True,
        commit=commit_val,
        describe=desc_val,
        dirty=dirty_val,
        error=err_commit,
    )

