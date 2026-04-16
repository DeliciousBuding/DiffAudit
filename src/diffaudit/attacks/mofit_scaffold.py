from __future__ import annotations

import json
from pathlib import Path


def build_mofit_summary(
    *,
    target_family: str,
    caption_source: str,
    surrogate_steps: int,
    embedding_steps: int,
    member_count: int,
    nonmember_count: int,
) -> dict:
    return {
        "task": "mofit-interface-canary",
        "status": "scaffold_only",
        "target_family": target_family,
        "caption_source": caption_source,
        "surrogate_steps": surrogate_steps,
        "embedding_steps": embedding_steps,
        "member_count": member_count,
        "nonmember_count": nonmember_count,
        "artifact_schema_version": "mofit-interface-canary.v1",
        "notes": (
            "Dedicated MoFit scaffold initialized. "
            "This artifact does not run surrogate or fitted-embedding optimization yet."
        ),
        "next_step": "implement surrogate and embedding optimization loops",
    }


def initialize_mofit_scaffold(
    *,
    run_root: Path,
    target_family: str,
    caption_source: str,
    surrogate_steps: int,
    embedding_steps: int,
    member_count: int,
    nonmember_count: int,
) -> dict[str, str]:
    run_root.mkdir(parents=True, exist_ok=True)
    traces_root = run_root / "traces"
    surrogate_trace_dir = traces_root / "surrogate"
    embedding_trace_dir = traces_root / "embedding"
    surrogate_trace_dir.mkdir(parents=True, exist_ok=True)
    embedding_trace_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_root / "summary.json"
    records_path = run_root / "records.jsonl"

    summary = build_mofit_summary(
        target_family=target_family,
        caption_source=caption_source,
        surrogate_steps=surrogate_steps,
        embedding_steps=embedding_steps,
        member_count=member_count,
        nonmember_count=nonmember_count,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    records_path.write_text("", encoding="utf-8")

    return {
        "summary_path": summary_path.as_posix(),
        "records_path": records_path.as_posix(),
        "surrogate_trace_dir": surrogate_trace_dir.as_posix(),
        "embedding_trace_dir": embedding_trace_dir.as_posix(),
    }
