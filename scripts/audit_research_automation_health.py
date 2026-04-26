from __future__ import annotations

import argparse
import json
from pathlib import Path

from diffaudit.research_automation_health import audit_research_automation_health


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit autonomous research workflow health from roadmap and markdown signals.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Research repo root.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the audit summary JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit_research_automation_health(args.repo_root)
    payload = json.dumps(summary, indent=2, ensure_ascii=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
