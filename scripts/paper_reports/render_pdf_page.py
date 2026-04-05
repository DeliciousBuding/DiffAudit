from __future__ import annotations

import argparse
from pathlib import Path

import fitz


def render_page(pdf_path: Path, page_number: int, output_path: Path, dpi: int) -> None:
    if page_number < 1:
        raise ValueError("page_number must be >= 1")

    document = fitz.open(pdf_path)
    try:
        page = document.load_page(page_number - 1)
        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pixmap.save(output_path)
    finally:
        document.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render one PDF page to a PNG file.",
    )
    parser.add_argument("pdf", type=Path, help="Input PDF path")
    parser.add_argument("page", type=int, help="1-based page number")
    parser.add_argument("output", type=Path, help="Output PNG path")
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Render resolution in DPI (default: 180)",
    )
    args = parser.parse_args()

    render_page(args.pdf, args.page, args.output, args.dpi)


if __name__ == "__main__":
    main()
