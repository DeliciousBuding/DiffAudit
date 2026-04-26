from __future__ import annotations

import argparse
import json
from pathlib import Path

from diffaudit.attacks.clid_threshold_audit import audit_clid_threshold_compatibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether a local CLiD run is close to released threshold-evaluator compatibility.")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--prepared-run", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_clid_threshold_compatibility(
        source_run=args.source_run,
        workspace=args.workspace,
        prepared_run=args.prepared_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
