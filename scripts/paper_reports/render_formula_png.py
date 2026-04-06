from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


PDFLATEX = shutil.which("pdflatex") or "pdflatex"
PDFTOCAIRO = shutil.which("pdftocairo") or "pdftocairo"


TEMPLATE = r"""
\documentclass[preview,border=6pt]{standalone}
\usepackage{amsmath}
\usepackage{amssymb}
\begin{document}
\[
%s
\]
\end{document}
"""


def render_formula(formula: str, output_path: Path, dpi: int) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        tex_path = tmp_dir / "formula.tex"
        pdf_path = tmp_dir / "formula.pdf"
        tex_path.write_text(TEMPLATE % formula, encoding="utf-8")

        subprocess.run(
            [
                PDFLATEX,
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={tmp_dir}",
                str(tex_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_stem = output_path.with_suffix("")
        subprocess.run(
            [
                PDFTOCAIRO,
                "-png",
                "-singlefile",
                "-r",
                str(dpi),
                str(pdf_path),
                str(output_stem),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a display LaTeX formula to PNG.")
    parser.add_argument("formula", nargs="?", help="LaTeX math expression without delimiters")
    parser.add_argument("output", type=Path, help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=360, help="PNG DPI (default: 360)")
    parser.add_argument(
        "--formula-file",
        type=Path,
        default=None,
        help="Read LaTeX formula content from a UTF-8 text file",
    )
    args = parser.parse_args()

    formula = args.formula
    if args.formula_file is not None:
        formula = args.formula_file.read_text(encoding="utf-8").strip()
    if not formula:
        raise SystemExit("formula text is required")

    render_formula(formula, args.output, args.dpi)


if __name__ == "__main__":
    main()
