from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from PIL import Image


FALLBACK_FINALIZE_TEMPLATE = Path(
    r"D:\Code\DiffAudit\Research\workspaces\black-box\runs\clid-recon-clip-target100-20260415-r1\finalize_run_summary.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a sanitized local CLiD probe from an existing prepared run.")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--target-run", type=Path, required=True)
    parser.add_argument("--jpeg-quality", type=int, default=70)
    parser.add_argument("--resize-down", type=int, default=448)
    parser.add_argument("--subset-limit", type=int, default=32)
    parser.add_argument("--max-n-samples", type=int, default=1)
    return parser.parse_args()


def load_run_config(run_root: Path) -> dict:
    return json.loads((run_root / "config.json").read_text(encoding="utf-8"))


def sanitize_image(src_path: Path, dst_path: Path, *, jpeg_quality: int, resize_down: int) -> None:
    image = Image.open(src_path).convert("RGB")
    width, height = image.size
    if resize_down > 0 and min(width, height) > resize_down:
        image = image.resize((resize_down, resize_down), Image.Resampling.BICUBIC)
        image = image.resize((width, height), Image.Resampling.BICUBIC)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    buffer.seek(0)
    sanitized = Image.open(buffer).convert("RGB")
    sanitized.save(dst_path, format="PNG")


def sanitize_split(src_dir: Path, dst_dir: Path, *, jpeg_quality: int, resize_down: int, subset_limit: int) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_metadata = src_dir / "metadata.jsonl"
    rows = src_metadata.read_text(encoding="utf-8").splitlines()
    kept_rows: list[str] = []
    count = 0
    for row in rows[:subset_limit]:
        payload = json.loads(row)
        file_name = payload["file_name"]
        sanitize_image(
            src_dir / file_name,
            dst_dir / file_name,
            jpeg_quality=jpeg_quality,
            resize_down=resize_down,
        )
        kept_rows.append(json.dumps(payload, ensure_ascii=False))
        count += 1
    (dst_dir / "metadata.jsonl").write_text("\n".join(kept_rows), encoding="utf-8")
    return count


def localize_script(
    script_text: str,
    *,
    use_data_model_name: str,
    member_dir: Path,
    nonmember_dir: Path,
    output_dir: Path,
    max_n_samples: int,
) -> str:
    text = script_text
    text = text.replace('Use_data_model_name = "local_paper_align_target"', f'Use_data_model_name = "{use_data_model_name}"')
    text = text.replace(
        '"local_paper_align_target": r"D:/Code/DiffAudit/Download/shared/weights/stable-diffusion-v1-5"',
        f'"{use_data_model_name}": r"D:/Code/DiffAudit/Download/shared/weights/stable-diffusion-v1-5"',
    )
    text = text.replace(
        '"local_paper_align_target": r"D:/Code/DiffAudit/Research/workspaces/black-box/runs/clid-paper-align-asset-prep-20260415-r1/datasets/member"',
        f'"{use_data_model_name}": r"{member_dir.as_posix()}"',
    )
    text = text.replace(
        '"local_paper_align_target": r"D:/Code/DiffAudit/Research/workspaces/black-box/runs/clid-paper-align-asset-prep-20260415-r1/datasets/nonmember"',
        f'"{use_data_model_name}": r"{nonmember_dir.as_posix()}"',
    )
    text = text.replace(
        "flags.outdir = r'D:/Code/DiffAudit/Research/workspaces/black-box/runs/clid-paper-align-asset-prep-20260415-r1/outputs'",
        f"flags.outdir = r'{output_dir.as_posix()}'",
    )
    text = text.replace("for Max_n_samples in [1]:", f"for Max_n_samples in [{max_n_samples}]:")
    return text


def main() -> None:
    args = parse_args()
    source_run = args.source_run
    target_run = args.target_run
    target_run.mkdir(parents=True, exist_ok=True)
    output_dir = target_run / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets_root = target_run / "datasets"
    member_dir = datasets_root / "member"
    nonmember_dir = datasets_root / "nonmember"

    config = load_run_config(source_run)
    source_member = Path(config["assets"]["member_dataset"])
    source_nonmember = Path(config["assets"]["nonmember_dataset"])

    member_count = sanitize_split(
        source_member,
        member_dir,
        jpeg_quality=args.jpeg_quality,
        resize_down=args.resize_down,
        subset_limit=args.subset_limit,
    )
    nonmember_count = sanitize_split(
        source_nonmember,
        nonmember_dir,
        jpeg_quality=args.jpeg_quality,
        resize_down=args.resize_down,
        subset_limit=args.subset_limit,
    )

    localized_script = localize_script(
        (source_run / "mia_CLiD_clip_local.py").read_text(encoding="utf-8"),
        use_data_model_name="local_sanitized_target",
        member_dir=member_dir,
        nonmember_dir=nonmember_dir,
        output_dir=output_dir,
        max_n_samples=args.max_n_samples,
    )
    (target_run / "mia_CLiD_clip_local.py").write_text(localized_script, encoding="utf-8")

    finalize_template = source_run / "finalize_run_summary.py"
    if not finalize_template.exists():
        finalize_template = FALLBACK_FINALIZE_TEMPLATE
    finalize_script = finalize_template.read_text(encoding="utf-8")
    finalize_script = finalize_script.replace(
        'RUN_ROOT = Path(\n    r"D:\\Code\\DiffAudit\\Research\\workspaces\\black-box\\runs\\clid-recon-clip-target100-20260415-r1"\n)',
        'RUN_ROOT = Path(\n'
        f'    r"{target_run.as_posix().replace("/", "\\\\")}"\n'
        ')',
    )
    finalize_script = finalize_script.replace(
        '"run_id": "clid-recon-clip-target100-20260415-r1",',
        f'"run_id": "{target_run.name}",',
    )
    finalize_script = finalize_script.replace(
        '"notes": (\n            "Local CLiD clip rung completed on 100 member + 100 non-member CelebA target samples using "\n            "base SD1.5 snapshot plus Recon target UNet attention-processor LoRA."\n        ),',
        '"notes": (\n'
        f'            "Local CLiD clip mitigation probe completed on sanitized served-image copies with JPEG quality {args.jpeg_quality} "\n'
        f'            f"and resize {args.resize_down} using the same staged SD1.5 base and Recon target LoRA."\n'
        '        ),',
    )
    (target_run / "finalize_run_summary.py").write_text(finalize_script, encoding="utf-8")

    prepared_config = {
        "run_id": target_run.name,
        "track": "black-box",
        "method": "clid",
        "status": "prepared",
        "mode": "mitigation-probe-local-bridge",
        "notes": "Prepared sanitized served-image probe for the local CLiD clip bridge.",
        "assets": {
            **config["assets"],
            "member_dataset": member_dir.as_posix(),
            "nonmember_dataset": nonmember_dir.as_posix(),
            "source_run": source_run.as_posix(),
        },
        "mitigation": {
            "name": "served-image-sanitization",
            "jpeg_quality": int(args.jpeg_quality),
            "resize_down": int(args.resize_down),
            "subset_limit": int(args.subset_limit),
        },
        "export": {
            "member_count": member_count,
            "nonmember_count": nonmember_count,
        },
    }
    (target_run / "config.json").write_text(json.dumps(prepared_config, indent=2, ensure_ascii=False), encoding="utf-8")
    (target_run / "analysis.md").write_text(
        "\n".join(
            [
                f"## {target_run.name}",
                "",
                "- Purpose: served-image-sanitization probe for the local CLiD bridge.",
                f"- Source run: `{source_run.as_posix()}`",
                f"- JPEG quality: `{args.jpeg_quality}`",
                f"- Resize-down: `{args.resize_down}`",
                f"- Subset per split: `{args.subset_limit}`",
            ]
        ),
        encoding="utf-8",
    )
    print(target_run)


if __name__ == "__main__":
    main()
