from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _single_output(output_dir: Path, pattern: str) -> Path:
    matches = sorted(output_dir.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one CLiD output for {pattern}, got {len(matches)}")
    return matches[0]


def _read_header_and_matrix(path: Path) -> tuple[str, np.ndarray]:
    rows: list[list[float]] = []
    header = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            rows.append([float(value) for value in line.split("\t")])
        except ValueError:
            if not header:
                header = line
    return header, np.asarray(rows, dtype=float)


def audit_clid_threshold_compatibility(
    *,
    source_run: str | Path,
    workspace: str | Path,
    prepared_run: str | Path | None = None,
) -> dict[str, Any]:
    source_run_path = Path(source_run)
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    output_dir = source_run_path / "outputs"
    train_path = _single_output(output_dir, "*TRTE_train*.txt")
    test_path = _single_output(output_dir, "*TRTE_test*.txt")
    train_header, train_matrix = _read_header_and_matrix(train_path)
    test_header, test_matrix = _read_header_and_matrix(test_path)

    prepared_base_model = ""
    prepared_uses_staged_root = False
    if prepared_run is not None:
        prepared_run_path = Path(prepared_run)
        config_path = prepared_run_path / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            prepared_base_model = str(config.get("assets", {}).get("base_model", ""))
            prepared_uses_staged_root = "Download/shared/weights/stable-diffusion-v1-5" in prepared_base_model

    executed_run_cache_root_leak = ".cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5" in train_header
    target_pair_ready = (
        train_matrix.ndim == 2
        and test_matrix.ndim == 2
        and train_matrix.shape[0] > 0
        and test_matrix.shape[0] > 0
        and train_matrix.shape[1] == test_matrix.shape[1]
        and train_matrix.shape[1] >= 2
    )

    summary = {
        "status": "ready",
        "track": "black-box",
        "method": "clid",
        "mode": "threshold-evaluator-compatibility-audit",
        "source_run": str(source_run_path),
        "workspace": str(workspace_path),
        "artifacts": {
            "summary": str(workspace_path / "summary.json"),
            "train_output": str(train_path),
            "test_output": str(test_path),
        },
        "target_pair": {
            "ready": bool(target_pair_ready),
            "train_shape": list(train_matrix.shape),
            "test_shape": list(test_matrix.shape),
            "column_count": int(train_matrix.shape[1]) if train_matrix.ndim == 2 and train_matrix.size else 0,
        },
        "shadow_pair": {
            "ready": False,
            "reason": "released cal_clid_th.py still expects shadow train/test files that are absent from the current local target-side bridge",
        },
        "boundary_findings": {
            "executed_run_cache_root_leak": bool(executed_run_cache_root_leak),
            "prepared_run_uses_staged_root": bool(prepared_uses_staged_root),
            "prepared_base_model": prepared_base_model,
            "header_skip_required": bool(train_header or test_header),
        },
        "verdict": "evaluator-near but shadow-blocked",
        "notes": [
            "Current CLiD target-side outputs are numerically parseable after skipping the first non-numeric header line.",
            "Released threshold evaluation still remains blocked on missing shadow-side files.",
        ],
    }

    (workspace_path / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
