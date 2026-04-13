"""Write StyleGAN3-compatible dataset.json labels for imagefolder directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, help="imagefolder dataset directory")
    return parser.parse_args()


def _image_files(dataset_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    )


def _label_for_file(image_path: Path) -> int:
    return int(image_path.stem.split("-", 1)[0])


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    image_paths = _image_files(dataset_dir)
    labels = []
    for image_path in image_paths:
        rel_path = image_path.relative_to(dataset_dir).as_posix()
        labels.append([rel_path, _label_for_file(image_path)])

    payload = {"labels": labels}
    output_path = dataset_dir / "dataset.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print({"dataset_dir": str(dataset_dir), "image_count": len(image_paths), "output": str(output_path)})


if __name__ == "__main__":
    main()
