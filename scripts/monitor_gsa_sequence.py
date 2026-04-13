"""Summarize the current status of a local GSA training/runtime sequence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CHECKPOINT_DIRS = ("target", "shadow-01", "shadow-02", "shadow-03")
EPOCH_RE = re.compile(r"Epoch\s+(\d+)")
STEP_RE = re.compile(r"step=(\d+)")
LOSS_RE = re.compile(r"loss=([0-9.]+)")


def _latest_checkpoints(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(item.name for item in path.iterdir() if item.is_dir() and item.name.startswith("checkpoint-"))


def _parse_training_log(log_path: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "exists": log_path.exists(),
        "latest_epoch": None,
        "latest_step": None,
        "latest_loss": None,
        "tail": [],
    }
    if not log_path.exists():
        return payload

    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    payload["tail"] = lines[-10:]
    for line in reversed(lines):
        if payload["latest_epoch"] is None:
            match = EPOCH_RE.search(line)
            if match:
                payload["latest_epoch"] = int(match.group(1))
        if payload["latest_step"] is None:
            match = STEP_RE.search(line)
            if match:
                payload["latest_step"] = int(match.group(1))
        if payload["latest_loss"] is None:
            match = LOSS_RE.search(line)
            if match:
                payload["latest_loss"] = float(match.group(1))
        if (
            payload["latest_epoch"] is not None
            and payload["latest_step"] is not None
            and payload["latest_loss"] is not None
        ):
            break
    return payload


def monitor_gsa_sequence(assets_root: str | Path, runtime_workspace: str | Path | None = None) -> dict[str, object]:
    assets_path = Path(assets_root)
    checkpoints_root = assets_path / "checkpoints"
    per_split: dict[str, object] = {}
    active_split = None

    for split in CHECKPOINT_DIRS:
        split_root = checkpoints_root / split
        log_path = split_root / "train.stderr.log"
        split_payload = {
            "path": str(split_root),
            "exists": split_root.exists(),
            "checkpoints": _latest_checkpoints(split_root),
            "training_log": _parse_training_log(log_path),
        }
        per_split[split] = split_payload
        if active_split is None and split_payload["training_log"]["latest_step"] is not None:
            active_split = split

    runtime_payload = {
        "workspace": "",
        "summary_exists": False,
        "summary_path": "",
    }
    if runtime_workspace:
        workspace_path = Path(runtime_workspace)
        summary_path = workspace_path / "summary.json"
        runtime_payload = {
            "workspace": str(workspace_path),
            "summary_exists": summary_path.exists(),
            "summary_path": str(summary_path),
        }

    if runtime_payload["summary_exists"]:
        phase = "runtime-ready"
    elif active_split is not None:
        phase = f"training-{active_split}"
    else:
        phase = "not-started"

    return {
        "status": "ready",
        "assets_root": str(assets_path),
        "phase": phase,
        "active_split": active_split,
        "splits": per_split,
        "runtime": runtime_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", required=True)
    parser.add_argument("--runtime-workspace", default="")
    args = parser.parse_args()

    payload = monitor_gsa_sequence(
        assets_root=args.assets_root,
        runtime_workspace=args.runtime_workspace or None,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
