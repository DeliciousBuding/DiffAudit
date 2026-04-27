from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = REPO_ROOT / "external" / "CLiD" / "mia_CLiD_clip.py"
DEFAULT_ASSET_CONFIG = REPO_ROOT / "configs" / "assets" / "staged-downloads.local.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a local CLiD bridge run using staged shared assets.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--member-pkl", type=Path, required=True)
    parser.add_argument("--nonmember-pkl", type=Path, required=True)
    parser.add_argument("--lora-dir", type=Path, required=True)
    parser.add_argument("--asset-config", type=Path, default=DEFAULT_ASSET_CONFIG)
    parser.add_argument("--template-script", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--use-data-model-name", type=str, default="local_paper_align")
    parser.add_argument("--export-limit", type=int, default=100)
    parser.add_argument("--max-n-samples", type=int, default=1)
    return parser.parse_args()


def load_asset_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def export_dataset(src_pkl: Path, out_dir: Path, limit: int) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Recon dataset.pkl stores PIL images, so torch 2.6+ needs weights_only=False here.
    obj = torch.load(src_pkl, map_location="cpu", weights_only=False)
    images = obj["image"][:limit]
    texts = obj["text"][:limit]
    metadata_rows: list[str] = []
    for idx, (image, text) in enumerate(zip(images, texts)):
        filename = f"{idx:03d}.png"
        image.save(out_dir / filename)
        metadata_rows.append(json.dumps({"file_name": filename, "text": text}, ensure_ascii=False))
    (out_dir / "metadata.jsonl").write_text("\n".join(metadata_rows), encoding="utf-8")
    return {
        "count": len(texts),
        "sample_text": texts[0] if texts else "",
    }


def localize_clid_clip(
    template_text: str,
    *,
    use_data_model_name: str,
    base_model: str,
    lora_dir: str,
    member_dir: Path,
    nonmember_dir: Path,
    output_dir: Path,
    max_n_samples: int,
) -> str:
    text = template_text
    text = text.replace(
        'Use_data_model_name = "flick_real_split1"',
        f'Use_data_model_name = "{use_data_model_name}"',
    )
    text = text.replace(
        'diff_path = {\n   "xx":"xx"\n}[Use_data_model_name]',
        'diff_path = {\n'
        f'   "{use_data_model_name}": r"{base_model}"\n'
        "}[Use_data_model_name]",
    )
    text = text.replace(
        'train_data_dict = {\n    "xx": "xx"\n\n}',
        'train_data_dict = {\n'
        f'    "{use_data_model_name}": r"{member_dir.as_posix()}"\n\n'
        "}",
    )
    text = text.replace(
        'test_data_dict = {\n    "xx": "xx"\n}',
        'test_data_dict = {\n'
        f'    "{use_data_model_name}": r"{nonmember_dir.as_posix()}"\n'
        "}",
    )
    text = text.replace(
        "print('unet loaded.')",
        "print('unet loaded.')\n"
        "import os\n"
        f"lora_dir = r'{lora_dir}'\n"
        "if os.path.isdir(lora_dir):\n"
        "    unet.load_attn_procs(lora_dir)\n"
        "    print('unet attn procs loaded from', lora_dir)",
    )
    text = text.replace("flags.train_batch_size = 8", "flags.train_batch_size = 1")
    text = text.replace("flags.outdir = 'outputs'", f"flags.outdir = r'{output_dir.as_posix()}'")
    text = text.replace("for Max_n_samples in [3, ]:  # 5, 7, 9]:", f"for Max_n_samples in [{max_n_samples}]:")
    text = text.replace("Noise = torch.randn(5000 * 40, 4, 64, 64)", "Noise = torch.randn(512, 4, 64, 64)")
    text = text.replace("subset_indices = range(2500)", "subset_indices = range(min(2500, len(test_dataset)))")
    text = text.replace("if flags.attack == 'mydenoise':", "if flags.attack in ['mydenoise', 'clid_clip']:")
    old = """        dataset = load_dataset(
            dataset_name,
            flags.dataset_config_name,
            cache_dir=flags.cache_dir,
            data_dir=flags.train_data_dir,
        )"""
    new = """        if os.path.isdir(dataset_name):
            dataset = load_dataset(
                "imagefolder",
                data_dir=dataset_name,
                cache_dir=flags.cache_dir,
            )
        else:
            dataset = load_dataset(
                dataset_name,
                flags.dataset_config_name,
                cache_dir=flags.cache_dir,
                data_dir=flags.train_data_dir,
            )"""
    return text.replace(old, new)


def main() -> None:
    args = parse_args()
    assets = load_asset_config(args.asset_config)
    base_model = assets["shared"]["sd15_model_dir"]
    clip_model_dir = assets["shared"].get("clip_model_dir")
    supplementary_root = assets.get("black_box", {}).get("clid", {}).get("supplementary_root")

    run_root = args.run_root
    run_root.mkdir(parents=True, exist_ok=True)
    datasets_root = run_root / "datasets"
    member_dir = datasets_root / "member"
    nonmember_dir = datasets_root / "nonmember"
    output_dir = run_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    member_info = export_dataset(args.member_pkl, member_dir, args.export_limit)
    nonmember_info = export_dataset(args.nonmember_pkl, nonmember_dir, args.export_limit)

    template_text = args.template_script.read_text(encoding="utf-8")
    localized = localize_clid_clip(
        template_text,
        use_data_model_name=args.use_data_model_name,
        base_model=base_model,
        lora_dir=args.lora_dir.as_posix(),
        member_dir=member_dir,
        nonmember_dir=nonmember_dir,
        output_dir=output_dir,
        max_n_samples=args.max_n_samples,
    )
    localized_script = run_root / "mia_CLiD_clip_local.py"
    localized_script.write_text(localized, encoding="utf-8")

    config = {
        "run_id": run_root.name,
        "track": "black-box",
        "method": "clid",
        "status": "prepared",
        "mode": "paper-alignment-local-bridge",
        "notes": (
            "Prepared a reusable local CLiD bridge using staged shared SD1.5 assets instead of a user Hugging Face "
            "cache snapshot. This is a paper-alignment hygiene step, not yet a calibrated benchmark verdict."
        ),
        "assets": {
            "asset_config": args.asset_config.as_posix(),
            "template_script": args.template_script.as_posix(),
            "base_model": base_model,
            "clip_model_dir": clip_model_dir,
            "supplementary_root": supplementary_root,
            "lora_dir": args.lora_dir.as_posix(),
            "member_pkl": args.member_pkl.as_posix(),
            "nonmember_pkl": args.nonmember_pkl.as_posix(),
            "member_dataset": member_dir.as_posix(),
            "nonmember_dataset": nonmember_dir.as_posix(),
        },
        "export": {
            "member": member_info,
            "nonmember": nonmember_info,
        },
    }
    (run_root / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_root / "analysis.md").write_text(
        "\n".join(
            [
                f"## {run_root.name}",
                "",
                "- Purpose: replace user-cache SD1.5 dependency with staged shared asset pointer.",
                "- Boundary: prepared local bridge only; no new benchmark verdict yet.",
                f"- Base model: `{base_model}`",
                f"- CLIP root: `{clip_model_dir}`",
                f"- Supplementary root: `{supplementary_root}`",
                f"- LoRA dir: `{args.lora_dir.as_posix()}`",
            ]
        ),
        encoding="utf-8",
    )
    print(run_root)


if __name__ == "__main__":
    main()
