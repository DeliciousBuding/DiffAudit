# DiffAudit Research Download Master List

> Root download directory: `D:\Code\DiffAudit\Download`
> Principle: top-level only grouped by `black-box / gray-box / white-box / shared`
> GitHub repositories are intentionally excluded because the agent can fetch those itself.
> Current local/manual status snapshot: `D:\Code\DiffAudit\Research\docs\research-download-current-status.md`

---

## 1. Directory Rules

Download all assets into these directories:

- `D:\Code\DiffAudit\Download\black-box\datasets`
- `D:\Code\DiffAudit\Download\black-box\weights`
- `D:\Code\DiffAudit\Download\black-box\papers`
- `D:\Code\DiffAudit\Download\black-box\supplementary`
- `D:\Code\DiffAudit\Download\gray-box\datasets`
- `D:\Code\DiffAudit\Download\gray-box\weights`
- `D:\Code\DiffAudit\Download\gray-box\papers`
- `D:\Code\DiffAudit\Download\gray-box\supplementary`
- `D:\Code\DiffAudit\Download\white-box\datasets`
- `D:\Code\DiffAudit\Download\white-box\weights`
- `D:\Code\DiffAudit\Download\white-box\papers`
- `D:\Code\DiffAudit\Download\white-box\supplementary`
- `D:\Code\DiffAudit\Download\shared\datasets`
- `D:\Code\DiffAudit\Download\shared\weights`
- `D:\Code\DiffAudit\Download\shared\papers`
- `D:\Code\DiffAudit\Download\shared\supplementary`

Machine-readable mirror:

- `D:\Code\DiffAudit\Download\manifests\research-download-manifest.json`

---

## 2. Shared Assets

These assets support multiple directions and should be downloaded first.

| ID | Asset | Why We Need It | Source | Login / Gated | Est. Size | Download To |
|---|---|---|---|---|---|---|
| `SH-DS-01` | CIFAR-10 python archive | Common source dataset for gray-box and white-box pipelines | [cs.toronto.edu CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) | No | ~170 MB | `D:\Code\DiffAudit\Download\shared\datasets\cifar-10-python.tar.gz` |
| `SH-DS-02` | CelebA image archive | Needed for black-box and image-level transfer lines | [CelebA official page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | Usually manual agreement / mirror choice | ~1.3 GB | `D:\Code\DiffAudit\Download\shared\datasets\celeba\img_align_celeba.zip` |
| `SH-DS-03` | CelebA annotation files | Needed to rebuild splits / metadata when required | [CelebA official page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | Usually manual agreement / mirror choice | small | `D:\Code\DiffAudit\Download\shared\datasets\celeba\` |
| `SH-WT-01` | Stable Diffusion v1.5 base model | Required by CLiD bridge and any SD1.5-based black-box work | [HF stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | Yes, gated / license | ~4-7 GB | `D:\Code\DiffAudit\Download\shared\weights\stable-diffusion-v1-5\` |
| `SH-WT-02` | OpenAI CLIP ViT-L/14 | Core encoder for CLiD-style scoring | [HF openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) | No | ~1.7 GB | `D:\Code\DiffAudit\Download\shared\weights\clip-vit-large-patch14\` |
| `SH-WT-03` | BLIP image captioning large | Useful for reconstruction-based black-box variants and caption-side probes | [HF Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large) | No | ~2 GB | `D:\Code\DiffAudit\Download\shared\weights\blip-image-captioning-large\` |
| `SH-WT-04` | Google DDPM CIFAR-10 32 | Useful shared baseline checkpoint for diffusion experiments and compatibility checks | [HF google/ddpm-cifar10-32](https://huggingface.co/google/ddpm-cifar10-32) | No | ~300 MB | `D:\Code\DiffAudit\Download\shared\weights\google-ddpm-cifar10-32\` |
| `SH-WT-05` | OpenCLIP / LAION CLIP fallback | Optional fallback if CLiD or new black-box variants benefit from open-CLIP family tests | [HF laion/CLIP-ViT-L-14-laion2B-s32B-b82K](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K) | No | ~1.7 GB | `D:\Code\DiffAudit\Download\shared\weights\openclip-vit-l-14\` |
| `SH-DS-04` | LSUN subset archive(s) | Optional for dataset dependence / transfer experiments later | [LSUN page](https://www.yf.io/p/lsun) | Mirror / manual | very large | `D:\Code\DiffAudit\Download\shared\datasets\lsun\` |

---

## 3. Black-Box Assets

| ID | Asset | Why We Need It | Source | Login / Gated | Est. Size | Download To |
|---|---|---|---|---|---|---|
| `BB-SUP-01` | Recon official non-GitHub asset bundle or mirrors | Helpful if current local recon assets need to be rebuilt / cross-validated | manual author / project release if available | Possibly manual | unknown | `D:\Code\DiffAudit\Download\black-box\supplementary\recon-official-assets\` |
| `BB-SUP-02` | CLiD supplementary / artifact release | Helpful for paper-faithful upgrade beyond current local bridge | manual author / supplementary release | Possibly manual | unknown | `D:\Code\DiffAudit\Download\black-box\supplementary\clid-release\` |
| `BB-PAP-01` | Any missing black-box paper supplements | Backup for protocol details if local PDFs are insufficient | paper supplement pages | maybe manual | small | `D:\Code\DiffAudit\Download\black-box\papers\` |

### Already local, no urgent external download required

- Current `Recon` asset stack is already present in-repo under `external/recon-assets/`
- Current `CLiD` codebase is already present under `external/CLiD/`

---

## 4. Gray-Box Assets

| ID | Asset | Why We Need It | Source | Login / Gated | Est. Size | Download To |
|---|---|---|---|---|---|---|
| `GB-WT-01` | SecMI official CIFAR10/CIFAR100 checkpoint bundle | Already validated locally; keep external copy staged for rebuilds or refresh | [SecMI OneDrive bundle](https://drexel0-my.sharepoint.com/:f:/g/personal/jd3734_drexel_edu/EnVid-empkpNvzC_mOfHwv0BpgkDsB_C4RmHO4rIH8BSzw?e=c17NjE) | Yes, browser/manual | unknown | `D:\Code\DiffAudit\Download\gray-box\weights\secmi-cifar-bundle\` |
| `GB-SUP-01` | PIA upstream non-GitHub checkpoints or release artifacts | Useful if we need to tighten provenance or compare against upstream weights | author release / paper supplement if available | maybe manual | unknown | `D:\Code\DiffAudit\Download\gray-box\supplementary\pia-upstream-assets\` |
| `GB-PAP-01` | SIMA paper / supplement if local copy is missing details | Needed if we implement SIMA | paper page / supplement | maybe manual | small | `D:\Code\DiffAudit\Download\gray-box\papers\sima\` |
| `GB-PAP-02` | MoFit paper / supplement | Needed if we implement MoFit | OpenReview / supplementary | maybe manual | small | `D:\Code\DiffAudit\Download\gray-box\papers\mofit\` |
| `GB-PAP-03` | Noise-as-a-probe paper / supplement | Needed if we implement this line | arXiv / supplementary | no or manual | small | `D:\Code\DiffAudit\Download\gray-box\papers\noise-as-a-probe\` |
| `GB-PAP-04` | SIDe paper / supplement | Needed if we implement this line | arXiv / supplementary | no or manual | small | `D:\Code\DiffAudit\Download\gray-box\papers\side\` |
| `GB-PAP-05` | Structural memorization paper / supplement | Needed if we implement this line | arXiv / supplementary | no or manual | small | `D:\Code\DiffAudit\Download\gray-box\papers\structural-memorization\` |

### Already local, no urgent external download required

- `external/PIA/` already exists
- `external/SecMI/` already exists
- current CIFAR-based gray-box local assets already exist in `workspaces/gray-box/assets/...`

---

## 5. White-Box Assets

| ID | Asset | Why We Need It | Source | Login / Gated | Est. Size | Download To |
|---|---|---|---|---|---|---|
| `WB-SUP-01` | NeMo supplementary / released artifacts | Needed to turn NeMo into a real verdict if public artifact exists | paper supplement / author release | maybe manual | unknown | `D:\Code\DiffAudit\Download\white-box\supplementary\nemo\` |
| `WB-PAP-01` | Local Mirror paper / supplement | Needed if we evaluate Local Mirror seriously | paper page / supplement | maybe manual | small | `D:\Code\DiffAudit\Download\white-box\papers\local-mirror\` |
| `WB-WT-01` | Additional CIFAR/DDPM checkpoints for alternate white-box protocols | Useful for non-GSA white-box variants if current local checkpoints are too narrow | official release / HF / author release | maybe gated | unknown | `D:\Code\DiffAudit\Download\white-box\weights\alternate-ddpm-checkpoints\` |

### Already local, no urgent external download required

- current GSA-oriented code and local assets already exist in `workspaces/white-box/`

---

## 6. Defense-Relevant Assets

There is no separate top-level `defense` folder by request. Defense assets should be staged under the box they most naturally belong to:

- black-box mitigation assets → `black-box\...`
- gray-box defense assets → `gray-box\...`
- white-box defended assets → `white-box\...`
- reusable base models / datasets → `shared\...`

Most defense directions currently rely more on code and training than on special non-GitHub downloads.

High-probability external needs:

- DPDM released checkpoints or defensive artifacts, if any become available
- defense paper supplements with exact protocol details
- any non-GitHub released defended checkpoints for diffusion models

Recommended landing spots:

- `D:\Code\DiffAudit\Download\gray-box\supplementary\defenses\`
- `D:\Code\DiffAudit\Download\white-box\supplementary\defenses\`

---

## 7. What You Probably Should Download First

If you want to front-load the highest-value assets before we reopen serious experiments, do these first:

1. `SH-DS-01` CIFAR-10
2. `SH-DS-02` / `SH-DS-03` CelebA
3. `SH-WT-01` Stable Diffusion v1.5
4. `SH-WT-02` CLIP ViT-L/14
5. `SH-WT-03` BLIP large
6. `GB-WT-01` SecMI official bundle
7. `WB-SUP-01` / `BB-SUP-02` if you can obtain real supplementary artifacts

---

## 8. Storage Reality Check

You said you currently have about `200 GB` free and around `10 MB/s` bandwidth.

That means:

- core shared assets are feasible;
- one or two huge optional datasets like `LSUN` should be treated as optional, not first-wave mandatory;
- gated Hugging Face model downloads should be staged deliberately, not all at once.

My practical recommendation:

- first wave: CIFAR-10, CelebA, SD1.5, CLIP, BLIP, SecMI bundle
- second wave: optional open-CLIP fallback and alternate checkpoint families
- third wave: very large datasets / author-supplied extras / rare supplements

---

## 9. Already In Repo

These do **not** need you to re-download right now:

- GitHub code repos under `external/`
- local frozen competition package under `workspaces/implementation/`
- local gray-box and white-box run artifacts already staged in `workspaces/`
- papers already archived in `references/materials/`

If a future task needs them refreshed from upstream, I can tell you precisely which one.
