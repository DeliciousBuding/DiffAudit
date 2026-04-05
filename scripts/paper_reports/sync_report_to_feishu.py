from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


PUBLIC_PERMISSION_PAYLOAD = {
    "link_share_entity": "anyone_readable",
    "share_entity": "anyone",
    "external_access_entity": "open",
    "comment_entity": "anyone_can_view",
    "copy_entity": "anyone_can_view",
    "security_entity": "anyone_can_view",
}

LARK_CLI = shutil.which("lark-cli") or shutil.which("lark-cli.cmd") or "lark-cli"
PWSH = shutil.which("pwsh") or shutil.which("powershell") or "powershell"


def run_command(args: list[str], cwd: Path) -> dict:
    completed = subprocess.run(
        args,
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return json.loads(completed.stdout)


def run_markdown_command(
    cwd: Path,
    command: str,
    title: str,
    markdown: str,
    doc: str | None = None,
) -> dict:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".md",
        delete=False,
    ) as temp_file:
        temp_file.write(markdown)
        temp_path = Path(temp_file.name)

    escaped_title = title.replace("'", "''")
    escaped_doc = doc.replace("'", "''") if doc else ""
    escaped_lark = str(Path(LARK_CLI)).replace("'", "''")
    escaped_temp = str(temp_path).replace("'", "''")

    if command == "create":
        script = (
            f"$md = Get-Content -Raw -LiteralPath '{escaped_temp}'; "
            f"& '{escaped_lark}' docs +create --as user --title '{escaped_title}' --markdown $md"
        )
    elif command == "update":
        script = (
            f"$md = Get-Content -Raw -LiteralPath '{escaped_temp}'; "
            f"& '{escaped_lark}' docs +update --as user --doc '{escaped_doc}' --mode overwrite --markdown $md"
        )
    else:
        script = (
            f"$md = Get-Content -Raw -LiteralPath '{escaped_temp}'; "
            f"& '{escaped_lark}' docs +update --as user --doc '{escaped_doc}' --mode append --markdown $md"
        )

    try:
        completed = subprocess.run(
            [PWSH, "-NoLogo", "-NoProfile", "-Command", script],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return json.loads(completed.stdout)
    finally:
        temp_path.unlink(missing_ok=True)


def split_markdown(markdown: str, limit: int = 3000) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in markdown.splitlines(keepends=True):
        if current and current_len + len(line) > limit:
            chunks.append("".join(current).strip())
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += len(line)
    if current:
        chunks.append("".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def extract_doc_id(payload: dict) -> str:
    data = payload.get("data", {})
    doc_id = data.get("doc_id") or data.get("document_id")
    if doc_id:
        return doc_id
    url = data.get("url") or data.get("doc_url")
    if url:
        match = re.search(r"/docx/([A-Za-z0-9]+)", url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract doc_id from payload: {payload}")


def remove_local_markdown_images(markdown: str) -> str:
    return re.sub(r"(?m)^!\[[^\]]*\]\([^)]+\)\s*$", "", markdown)


def derive_title(report_path: Path, markdown: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return report_path.stem


def create_or_update_doc(
    cwd: Path,
    title: str,
    markdown: str,
    doc: str | None,
) -> str:
    chunks = split_markdown(markdown)
    if not chunks:
        raise ValueError("markdown is empty")

    if doc:
        payload = run_markdown_command(cwd, "update", title, chunks[0], doc)
        doc_id = extract_doc_id(payload)
        for chunk in chunks[1:]:
            run_markdown_command(
                cwd,
                "append",
                title,
                chunk,
                f"https://www.feishu.cn/docx/{doc_id}",
            )
        return doc_id

    payload = run_markdown_command(cwd, "create", title, chunks[0])
    doc_id = extract_doc_id(payload)
    for chunk in chunks[1:]:
        run_markdown_command(
            cwd,
            "append",
            title,
            chunk,
            f"https://www.feishu.cn/docx/{doc_id}",
        )
    return doc_id


def set_public_permission(cwd: Path, doc_id: str) -> None:
    run_command(
        [
            LARK_CLI,
            "api",
            "PATCH",
            f"/open-apis/drive/v2/permissions/{doc_id}/public",
            "--params",
            json.dumps({"type": "docx"}, ensure_ascii=True),
            "--data",
            json.dumps(PUBLIC_PERMISSION_PAYLOAD, ensure_ascii=True),
            "--as",
            "user",
        ],
        cwd,
    )


def insert_image(cwd: Path, doc_id: str, image_path: Path, caption: str | None) -> None:
    relative_path = Path(image_path).resolve().relative_to(cwd.resolve())
    args = [
        LARK_CLI,
        "docs",
        "+media-insert",
        "--as",
        "user",
        "--doc",
        f"https://www.feishu.cn/docx/{doc_id}",
        "--type",
        "image",
        "--file",
        str(relative_path),
    ]
    if caption:
        args.extend(["--caption", caption])
    run_command(args, cwd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or update a Feishu report doc from a local Markdown report.",
    )
    parser.add_argument("report", type=Path, help="Local Markdown report path")
    parser.add_argument(
        "--doc",
        type=str,
        default=None,
        help="Existing Feishu doc URL or doc_id to update",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional local image asset to insert after markdown upload",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional explicit Feishu doc title",
    )
    parser.add_argument(
        "--image-caption",
        type=str,
        default=None,
        help="Optional caption for the inserted image",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    markdown_raw = args.report.read_text(encoding="utf-8")
    markdown_upload = remove_local_markdown_images(markdown_raw)
    title = args.title or derive_title(args.report, markdown_raw)

    doc_id = create_or_update_doc(cwd, title, markdown_upload, args.doc)
    set_public_permission(cwd, doc_id)

    if args.image:
        insert_image(cwd, doc_id, args.image, args.image_caption)

    print(
        json.dumps(
            {
                "doc_id": doc_id,
                "doc_url": f"https://www.feishu.cn/docx/{doc_id}",
                "title": title,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
