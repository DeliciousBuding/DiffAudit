from __future__ import annotations

import csv
import re
from pathlib import Path


def extract_h1(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def extract_summary_paragraphs(markdown: str) -> list[str]:
    match = re.search(
        r"## Extracted Summary for `paper-index\.md`\n(.*?)(?=\n## |\Z)",
        markdown,
        flags=re.S,
    )
    if not match:
        return []
    section = match.group(1).strip()
    return [part.strip() for part in re.split(r"\n\s*\n", section) if part.strip()]


def load_report_data(research_root: Path) -> dict[str, dict[str, str]]:
    manifest_path = research_root / "docs" / "paper-reports" / "manifest.csv"
    data: dict[str, dict[str, str]] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            material_path = row["material_path"]
            if not material_path.startswith("references/materials/"):
                continue
            relative_material = material_path.replace("references/materials/", "", 1)
            report_path = research_root / row["report_path"]
            if not report_path.exists():
                continue
            markdown = report_path.read_text(encoding="utf-8")
            paragraphs = extract_summary_paragraphs(markdown)
            if len(paragraphs) != 3:
                continue
            data[relative_material] = {
                "title": extract_h1(markdown) or report_path.stem,
                "doc_url": row.get("feishu_doc_url", ""),
                "p1": paragraphs[0],
                "p2": paragraphs[1],
                "p3": paragraphs[2],
            }
    return data


def update_paper_index(research_root: Path) -> None:
    paper_index_path = research_root / "references" / "materials" / "paper-index.md"
    report_data = load_report_data(research_root)
    lines = paper_index_path.read_text(encoding="utf-8").splitlines()
    new_lines: list[str] = []
    current_material: str | None = None

    reading_report_field_present = False

    for line in lines:
        if line.strip() == "- `阅读报告`：对应单篇详细阅读报告的飞书文档":
            reading_report_field_present = True

        file_match = re.match(r"- 文件：\[.*?\]\(([^)]+)\)", line)
        if file_match:
            current_material = file_match.group(1)
            new_lines.append(line)
            continue

        if current_material and current_material in report_data:
            report = report_data[current_material]
            if line.startswith("- 内容简介："):
                new_lines.append(f"- 内容简介：{report['p1']}")
                continue
            if line.startswith("- 核心方法 / 结论："):
                new_lines.append(f"- 核心方法 / 结论：{report['p2']}")
                continue
            if line.startswith("- 和 DiffAudit 的关系："):
                new_lines.append(f"- 和 DiffAudit 的关系：{report['p3']}")
                continue
            if line.startswith("- 阅读报告："):
                continue
            if line.startswith("- 开源仓库："):
                new_lines.append(line)
                if report["doc_url"]:
                    new_lines.append(f"- 阅读报告：[{report['title']}]({report['doc_url']})")
                continue

        new_lines.append(line)

    if not reading_report_field_present:
        insertion = "- `阅读报告`：对应单篇详细阅读报告的飞书文档"
        for idx, line in enumerate(new_lines):
            if line.strip() == "- `开源仓库`：优先给官方或作者公开实现；找不到就明确写“暂未找到”":
                new_lines.insert(idx + 1, insertion)
                break

    paper_index_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def main() -> None:
    research_root = Path(__file__).resolve().parents[2]
    update_paper_index(research_root)


if __name__ == "__main__":
    main()
