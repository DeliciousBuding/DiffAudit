# Data And Assets Handoff

这份文档回答新同学接手时最容易卡住的问题：

- 数据集、权重、supplementary 包到底从哪里来
- 下载后放到哪里
- 如何让本机路径和仓库命令对上
- 如何验证自己已经拿到和当前项目一致的资产布局

## 1. Canonical Layout

默认目录布局是：

```text
D:\Code\DiffAudit\
  Research\        # this git repo
  Download\        # local raw datasets / weights / supplementary bundles
```

`Download\` 不进 `Research` git 仓库。它是本机原始资产层，原因是这里会放大文件、许可受限文件、可重新下载的权重和压缩包。

如果你的机器不使用 `D:\Code\DiffAudit\`，也可以换成自己的根目录，但必须保持同样的相对角色：

- `Research\external\` 只放外部代码 clone
- `Research\third_party\` 只放最小 vendored 代码
- `Download\` 放原始下载物
- `Research\workspaces\<lane>\assets\` 放已归一化的 lane 入口
- `Research\workspaces\<lane>\runs\` 放运行证据

详细规则见 [storage-boundary.md](storage-boundary.md)。

## 2. Fastest Way To Match Our Assets

如果你能从项目机器或共享盘拿到完整资产目录，最快方式是直接复制整个目录：

```powershell
Copy-Item -Recurse \\YOUR_ASSET_MIRROR\DiffAudit\Download D:\Code\DiffAudit\Download
```

复制后确认至少有这些入口：

```text
Download\black-box\supplementary\recon-assets\
Download\black-box\supplementary\clid-mia-supplementary\
Download\gray-box\weights\secmi-cifar-bundle\
Download\shared\datasets\cifar-10-python.tar.gz
Download\shared\datasets\celeba\
Download\shared\weights\stable-diffusion-v1-5\
Download\shared\weights\clip-vit-large-patch14\
Download\shared\weights\blip-image-captioning-large\
Download\shared\weights\google-ddpm-cifar10-32\
```

如果不能复制，按 [research-download-master-list.md](research-download-master-list.md) 的 `first-wave` 顺序下载。部分资产有登录、许可或作者发布限制，仓库只能记录来源和目标目录，不能把所有大文件直接提交进 git。

## 3. First-Wave Asset List

新机器优先拿这些，足够进入当前项目的大多数接手和验证路径：

| Asset ID | What | Source | Destination |
|---|---|---|---|
| `SH-DS-01` | CIFAR-10 archive | https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz | `Download\shared\datasets\cifar-10-python.tar.gz` |
| `SH-DS-02/03` | CelebA images + annotations | https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html | `Download\shared\datasets\celeba\` |
| `SH-WT-01` | Stable Diffusion v1.5 | https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 | `Download\shared\weights\stable-diffusion-v1-5\` |
| `SH-WT-02` | CLIP ViT-L/14 | https://huggingface.co/openai/clip-vit-large-patch14 | `Download\shared\weights\clip-vit-large-patch14\` |
| `SH-WT-03` | BLIP large | https://huggingface.co/Salesforce/blip-image-captioning-large | `Download\shared\weights\blip-image-captioning-large\` |
| `SH-WT-04` | DDPM CIFAR-10 32 | https://huggingface.co/google/ddpm-cifar10-32 | `Download\shared\weights\google-ddpm-cifar10-32\` |
| `GB-WT-01` | SecMI CIFAR bundle | see [research-download-master-list.md](research-download-master-list.md) | `Download\gray-box\weights\secmi-cifar-bundle\` |
| `BB-SUP-02` | CLiD supplementary mirror | paper supplementary / manual bundle | `Download\black-box\supplementary\clid-mia-supplementary\` |

The current recon bundle is already staged on the project machine under:

```text
Download\black-box\supplementary\recon-assets\
```

If you do not have that directory, ask for the project asset mirror or rebuild it from the release/source noted in [recon-public-asset-mapping.md](recon-public-asset-mapping.md).

## 4. External Code Clones

These are code clones, not data directories. They belong under `Research\external\`:

```powershell
git clone https://github.com/kong13661/PIA.git external/PIA
git clone https://github.com/zhaisf/CLiD external/CLiD
git clone https://github.com/py85252876/Reconstruction-based-Attack external/Reconstruction-based-Attack
git clone https://github.com/facebookresearch/DiT.git external/DiT
git clone https://github.com/nv-tlabs/DPDM.git external/DPDM
```

`external/` is ignored by git. Do not put datasets or model weights there.

## 5. Bind Local Paths

After assets are present, create your local path config:

```powershell
Copy-Item configs/assets/team.local.template.yaml configs/assets/team.local.yaml
```

Fill these fields first:

```yaml
repo:
  research_root: D:/Code/DiffAudit/Research
  download_root: D:/Code/DiffAudit/Download

shared:
  cifar10_archive: D:/Code/DiffAudit/Download/shared/datasets/cifar-10-python.tar.gz
  celeba_root: D:/Code/DiffAudit/Download/shared/datasets/celeba
  sd15_model_dir: D:/Code/DiffAudit/Download/shared/weights/stable-diffusion-v1-5
  clip_model_dir: D:/Code/DiffAudit/Download/shared/weights/clip-vit-large-patch14
  blip_model_dir: D:/Code/DiffAudit/Download/shared/weights/blip-image-captioning-large
  ddpm_cifar10_model_dir: D:/Code/DiffAudit/Download/shared/weights/google-ddpm-cifar10-32
```

Then render local configs:

```powershell
python scripts/render_team_local_configs.py
```

`configs/assets/team.local.yaml` is ignored by git. Do not commit personal absolute paths.

## 6. Verification Commands

Run these after environment setup and local path binding:

```powershell
python scripts/verify_env.py
python -m diffaudit --help
python scripts/render_team_local_configs.py
```

Basic asset checks:

```powershell
conda run -n diffaudit-research python -m diffaudit probe-pia-assets --config configs/attacks/pia_plan.yaml --member-split-root external/PIA/DDPM
conda run -n diffaudit-research python -m diffaudit probe-gsa-assets --repo-root workspaces/white-box/external/GSA --assets-root workspaces/white-box/assets/gsa
conda run -n diffaudit-research python -m diffaudit audit-recon-public-bundle --bundle-root D:\Code\DiffAudit\Download\black-box\supplementary\recon-assets\ndss-2025-blackbox-membership-inference-fine-tuned-diffusion-models
```

Expected result:

- environment commands exit successfully
- `python -m diffaudit --help` lists all CLI commands
- asset probes either report `ready` or name the exact missing local path

## 7. If You Are Taking Over The Research Work

After the machine can run commands, read in this order:

1. [README.md](../README.md)
2. [teammate-setup.md](teammate-setup.md)
3. [comprehensive-progress.md](comprehensive-progress.md)
4. [reproduction-status.md](reproduction-status.md)
5. [mainline-narrative.md](mainline-narrative.md)
6. [storage-boundary.md](storage-boundary.md)
7. [research-download-master-list.md](research-download-master-list.md)

Current rule of thumb:

- use `Download\` to get the same raw data and weights
- use `team.local.yaml` to bind your machine paths
- use `workspaces/*/assets` manifests as the project-recognized asset contracts
- use `workspaces/*/runs` and implementation notes as evidence, not as raw dataset storage
