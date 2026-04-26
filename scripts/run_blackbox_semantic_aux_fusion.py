from __future__ import annotations

import argparse
import json
from pathlib import Path

from diffaudit.attacks.semantic_aux_fusion import (
    analyze_semantic_aux_fusion,
    load_semantic_aux_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate bounded fusion candidates on existing semantic-auxiliary-classifier record dumps."
    )
    parser.add_argument("--records", action="append", required=True, help="Path to outputs/records.json.")
    parser.add_argument("--run-root", type=Path, required=True, help="Directory to write summary artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)

    per_run: list[dict[str, object]] = []
    fusion_beats_mean_cos = False
    meaningful_gain = False
    for record_path in args.records:
        record_file = Path(record_path)
        analysis = analyze_semantic_aux_fusion(load_semantic_aux_records(record_file))
        analysis["records_path"] = record_file.as_posix()
        per_run.append(analysis)
        if analysis["best_candidate"] != "mean_cos":
            fusion_beats_mean_cos = True
        if float(analysis["best_auc_gain_vs_mean_cos"]) >= 0.01:
            meaningful_gain = True

    if not fusion_beats_mean_cos:
        verdict = "negative but useful"
        notes = (
            "No bounded fusion candidate beat mean_cos on either comparator; current semantic-aux ordering is still "
            "best understood as a mean_cos-led similarity signal."
        )
    elif not meaningful_gain:
        verdict = "negative but useful"
        notes = (
            "At least one bounded fusion variant edges past mean_cos, but the gain stays below the 0.01 AUC bar and "
            "does not justify a new challenger promotion."
        )
    else:
        verdict = "positive"
        notes = "A bounded fusion candidate clears the 0.01 AUC gain bar and should be considered for follow-up."

    summary = {
        "task_scope": "BB-2.2 bounded fusion experiments",
        "track": "black-box",
        "method_family": "semantic-auxiliary-classifier",
        "device": "cpu",
        "decision": verdict,
        "per_run": per_run,
        "overall": {
            "fusion_beats_mean_cos_any_run": fusion_beats_mean_cos,
            "meaningful_gain_ge_0p01_auc": meaningful_gain,
            "preferred_reference_score": "mean_cos" if verdict != "positive" else per_run[-1]["best_candidate"],
        },
        "notes": notes,
    }
    (args.run_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
