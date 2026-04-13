"""Rebuild canonical CIFAR-10 buckets for the GSA white-box line."""

from __future__ import annotations

import argparse
import pickle
import random
import tarfile
import tempfile
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-archive", required=True, help="path to cifar-10-python.tar.gz")
    parser.add_argument("--output-root", required=True, help="canonical GSA datasets root")
    parser.add_argument("--target-count", type=int, default=128, help="images per target split")
    parser.add_argument("--shadow-count", type=int, default=128, help="images per shadow split")
    parser.add_argument(
        "--shadow-pairs",
        type=int,
        default=1,
        help="number of independent shadow member/non-member split pairs",
    )
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    return parser.parse_args()


def _load_cifar_images(extracted_root: Path) -> list[tuple[str, object]]:
    batch_root = extracted_root / "cifar-10-batches-py"
    file_specs = [batch_root / f"data_batch_{index}" for index in range(1, 6)]
    items: list[tuple[str, object]] = []
    for file_path in file_specs:
        with file_path.open("rb") as handle:
            payload = pickle.load(handle, encoding="bytes")
        labels = payload[b"labels"]
        data = payload[b"data"]
        for item_index, row in enumerate(data):
            items.append((f"{labels[item_index]:02d}-{file_path.stem}-{item_index:05d}.png", row))
    return items


def _row_to_image(row: object) -> Image.Image:
    import numpy as np

    image = np.asarray(row, dtype=np.uint8).reshape(3, 32, 32).transpose(1, 2, 0)
    return Image.fromarray(image, mode="RGB")


def _write_bucket(bucket_dir: Path, items: list[tuple[str, object]]) -> None:
    bucket_dir.mkdir(parents=True, exist_ok=True)
    for existing in bucket_dir.glob("*"):
        if existing.is_file():
            existing.unlink()
    for filename, row in items:
        _row_to_image(row).save(bucket_dir / filename)


def _shadow_bucket_names(index: int, total_pairs: int) -> tuple[str, str]:
    if total_pairs == 1:
        return "shadow-member", "shadow-nonmember"
    prefix = f"shadow-{index:02d}"
    return f"{prefix}-member", f"{prefix}-nonmember"


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_root = Path(args.output_root)

    with tempfile.TemporaryDirectory(prefix="gsa-cifar-") as tmpdir:
        extracted_root = Path(tmpdir)
        with tarfile.open(args.source_archive, "r:gz") as archive:
            archive.extractall(extracted_root)

        items = _load_cifar_images(extracted_root)
        needed = args.target_count * 2 + args.shadow_pairs * args.shadow_count * 2
        if needed > len(items):
            raise SystemExit(f"requested {needed} images but archive only has {len(items)}")

        selected = rng.sample(items, needed)
        cursor = 0

        target_member = selected[cursor : cursor + args.target_count]
        cursor += args.target_count
        target_nonmember = selected[cursor : cursor + args.target_count]
        cursor += args.target_count
        _write_bucket(output_root / "target-member", target_member)
        _write_bucket(output_root / "target-nonmember", target_nonmember)
        shadow_summaries: list[dict[str, object]] = []
        for shadow_index in range(1, args.shadow_pairs + 1):
            shadow_member = selected[cursor : cursor + args.shadow_count]
            cursor += args.shadow_count
            shadow_nonmember = selected[cursor : cursor + args.shadow_count]
            cursor += args.shadow_count
            member_bucket_name, nonmember_bucket_name = _shadow_bucket_names(
                shadow_index, args.shadow_pairs
            )
            _write_bucket(output_root / member_bucket_name, shadow_member)
            _write_bucket(output_root / nonmember_bucket_name, shadow_nonmember)
            shadow_summaries.append(
                {
                    "index": shadow_index,
                    "member_bucket": member_bucket_name,
                    "nonmember_bucket": nonmember_bucket_name,
                    "member_count": len(shadow_member),
                    "nonmember_count": len(shadow_nonmember),
                }
            )

    summary = {
        "target_member": len(target_member),
        "target_nonmember": len(target_nonmember),
        "shadow_pairs": args.shadow_pairs,
        "shadow_count_per_split": args.shadow_count,
        "shadow_summaries": shadow_summaries,
        "output_root": str(output_root),
        "seed": args.seed,
    }
    print(summary)


if __name__ == "__main__":
    main()
