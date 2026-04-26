# 2026-04-15 First-Wave Download Verification

## Scope

This note verifies first-wave assets referenced by:

- `docs/research-download-master-list.md`
- `D:\Code\DiffAudit\Download\manifests\research-download-manifest.json`

It also records how newly downloaded assets are mapped into repo-consumable locations or local config pointers.

## Verification Summary

### Present and usable

| Asset ID | Local status | Verified path | Repo-consumable pointer |
|---|---|---|---|
| `SH-DS-01` | present | `D:\Code\DiffAudit\Download\shared\datasets\cifar-10-python.tar.gz` | `configs/assets/staged-downloads.local.yaml -> shared.cifar10_archive` |
| `SH-DS-02` | present | `D:\Code\DiffAudit\Download\shared\datasets\celeba\img_align_celeba.zip` | `configs/assets/staged-downloads.local.yaml -> shared.celeba_root` |
| `SH-DS-03` | present | `D:\Code\DiffAudit\Download\shared\datasets\celeba\` | `configs/assets/staged-downloads.local.yaml -> shared.celeba_root` |
| `SH-WT-01` | present | `D:\Code\DiffAudit\Download\shared\weights\stable-diffusion-v1-5\` | `configs/assets/staged-downloads.local.yaml -> shared.sd15_model_dir` |
| `SH-WT-02` | present | `D:\Code\DiffAudit\Download\shared\weights\clip-vit-large-patch14\` | `configs/assets/staged-downloads.local.yaml -> shared.clip_model_dir` |
| `SH-WT-03` | present | `D:\Code\DiffAudit\Download\shared\weights\blip-image-captioning-large\` | `configs/assets/staged-downloads.local.yaml -> shared.blip_model_dir` |
| `GB-WT-01` | present | `D:\Code\DiffAudit\Download\gray-box\weights\secmi-cifar-bundle\` | `workspaces/gray-box/assets/secmi/manifest.json` and `configs/assets/staged-downloads.local.yaml -> gray_box.secmi.*` |
| `BB-SUP-02` | present via local staging | `D:\Code\DiffAudit\Download\black-box\supplementary\clid-mia-supplementary\contents\CLID_MIA\` | `configs/assets/staged-downloads.local.yaml -> black_box.clid.supplementary_root` |

## Repo-Consumable Mapping Decisions

### Shared diffusion-model roots

- `SD1.5` should no longer depend on a user Hugging Face cache snapshot path.
- Canonical local pointer is now:
  - `D:\Code\DiffAudit\Download\shared\weights\stable-diffusion-v1-5\`
- `CLIP ViT-L/14` pointer is:
  - `D:\Code\DiffAudit\Download\shared\weights\clip-vit-large-patch14\`
- `BLIP large` pointer is:
  - `D:\Code\DiffAudit\Download\shared\weights\blip-image-captioning-large\`

### Gray-box SecMI

- Raw external archive retained under:
  - `D:\Code\DiffAudit\Download\gray-box\supplementary\secmi-onedrive\raw\`
- Extracted download bundle retained under:
  - `D:\Code\DiffAudit\Download\gray-box\weights\secmi-cifar-bundle\`
- Current repo-canonical executable root remains:
  - `workspaces/gray-box/assets/secmi/checkpoints/CIFAR10`
- Existing provenance artifact already covers the workspace staging:
  - `workspaces/gray-box/assets/secmi/manifest.json`
  - `workspaces/gray-box/assets/secmi/PROVENANCE.md`

### Black-box CLiD supplementary

- Newly downloaded supplementary code/artifacts are retained under:
  - `D:\Code\DiffAudit\Download\black-box\supplementary\clid-mia-supplementary\contents\CLID_MIA\`
- Current runnable local CLiD line still executes from:
  - `external/CLiD/`
- The supplementary bundle is therefore staged as:
  - paper-alignment support material
  - not as an automatic replacement of the current local bridge

## Verdict on P0 download tasks

- `P0-DL-1`:
  - satisfied after `SH-WT-03` (`BLIP large`) finished and the final `model.safetensors` landed under the staged shared weights root
- `P0-DL-2`:
  - repo-consumable pointer file prepared at `configs/assets/staged-downloads.local.yaml`
- `P0-DL-3`:
  - first-wave path mapping and staging logic recorded in this note

## Next Action

1. Wait for `BLIP large` to finish and re-run a short verification.
2. If complete, check `P0-DL-1`.
3. Then move directly to `P0-CL-1` using the new local pointers instead of user cache paths.
