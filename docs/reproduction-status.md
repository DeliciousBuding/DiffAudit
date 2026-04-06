# 复现状态总览

这份文档用于汇总仓库当前各条攻击线的真实推进状态。

判断标准统一分为：

- `research-ready`：论文、代码仓库、资产要求已整理
- `code-ready`：仓库内已有 planner / probe / dry-run / smoke 等实现
- `evidence-ready`：仓库内已经有可提交的 `summary.json` 或等价运行证据
- `asset-ready`：真实论文资产已到位，可以开始逼近真实 benchmark
- `benchmark-ready`：已经可以合理声称在跑或复算论文级结果

## 黑盒

| 方法 | 论文 | 当前状态 | 仓库命令 | 当前证据 | 主要阻塞 |
| --- | --- | --- | --- | --- | --- |
| `clid` | `2024-neurips-clid-membership-inference-text-to-image-diffusion.pdf` | `evidence-ready` | `plan-clid` / `probe-clid-assets` / `dry-run-clid` / `run-clid-dry-run-smoke` / `summarize-clid-artifacts` | [clid-dry-run-smoke](../experiments/clid-dry-run-smoke/summary.json), [clid-artifact-summary](../experiments/clid-artifact-summary/summary.json) | 缺真实 text-to-image 资产、缺可重复的目标模型路径与数据集映射 |
| `recon` | `2025-ndss-black-box-membership-inference-fine-tuned-diffusion-models.pdf` 与 `external/Reconstruction-based-Attack` | `code-ready + evidence-ready` | `plan-recon` / `probe-recon-assets` / `dry-run-recon` / `run-recon-mainline-smoke` / `probe-recon-runtime-assets` / `run-recon-runtime-mainline` / `probe-recon-score-artifacts` / `run-recon-artifact-mainline` / `run-recon-eval-smoke` / `summarize-recon-artifacts` / `run-recon-upstream-eval-smoke` | [recon-mainline-smoke](../experiments/recon-mainline-smoke/summary.json), [recon-eval-smoke](../experiments/recon-eval-smoke/summary.json), [recon-artifact-summary](../experiments/recon-artifact-summary/summary.json), [recon-upstream-eval-smoke](../experiments/recon-upstream-eval-smoke/summary.json), [recon-runtime-ddim-mainline](../experiments/recon-runtime-mainline-ddim-smoke/summary.json), [recon-runtime-ddim-artifact-mainline](../experiments/recon-runtime-mainline-ddim-smoke/artifact-mainline-final/summary.json), [recon-runtime-ddim-public-10](../experiments/recon-runtime-mainline-ddim-public-10-step10/summary.json), [recon-runtime-ddim-public-10-artifact-mainline](../experiments/recon-runtime-mainline-ddim-public-10-step10/artifact-mainline/summary.json), [recon-runtime-ddim-public-25](../experiments/recon-runtime-mainline-ddim-public-25-step10/summary.json), [recon-runtime-ddim-public-25-artifact-mainline](../experiments/recon-runtime-mainline-ddim-public-25-step10/artifact-mainline/summary.json), [recon-runtime-ddim-public-50](../experiments/recon-runtime-mainline-ddim-public-50-step10/summary.json), [recon-runtime-ddim-public-50-artifact-mainline](../experiments/recon-runtime-mainline-ddim-public-50-step10/artifact-mainline/summary.json), [recon-runtime-ddim-public-100](../experiments/recon-runtime-mainline-ddim-public-100-step10/summary.json), [recon-runtime-ddim-public-100-artifact-mainline](../experiments/recon-runtime-mainline-ddim-public-100-step10/artifact-mainline/summary.json), [recon-runtime-kandinsky-mainline](../experiments/recon-runtime-mainline-kandinsky-public-smoke/summary.json), [recon-runtime-kandinsky-artifact-mainline](../experiments/recon-runtime-mainline-kandinsky-public-smoke/artifact-mainline/summary.json) | 公开资产包（DOI: `10.5281/zenodo.13371475`）已经支撑 `Stable Diffusion + DDIM` 的 `100-sample public runtime-mainline` 与 `kandinsky_v22` 的最小真实 runtime-mainline；当前最大的公开子集证据是 `DDIM public-100 step10`，指标为 `auc=0.788 / asr=0.63 / tpr@1%fpr=0.99`，但当前最佳 AUC 仍来自 `DDIM public-50 step10` 的 `0.866`，主要阻塞转为公开资产语义映射核准与跨模型覆盖补齐 |
| `variation` | `2024-arxiv-towards-black-box-membership-inference-diffusion-models.pdf` 对应 API-only 路线 | `code-ready + evidence-ready` | `plan-variation` / `probe-variation-assets` / `dry-run-variation` / `run-variation-synth-smoke` | [variation-synth-smoke](../experiments/variation-synth-smoke/summary.json) | 缺真实 API 凭据、调用预算和真实 query image 集 |

## 灰盒

| 方法 | 论文 | 当前状态 | 仓库命令 | 当前证据 | 主要阻塞 |
| --- | --- | --- | --- | --- | --- |
| `secmi` | `2023-icml-secmi-membership-inference-diffusion-models.pdf` | `code-ready + evidence-ready` | `plan-secmi` / `probe-secmi-assets` / `prepare-secmi` / `dry-run-secmi` / `runtime-probe-secmi` / `run-secmi-synth-smoke` | [secmi-synth-smoke](../experiments/secmi-synth-smoke/summary.json), [secmi-synth-smoke-gpu](../experiments/secmi-synth-smoke-gpu/summary.json) | 缺真实 checkpoint、flagfile 和论文一致资产布局 |
| `pia` | `2024-iclr-pia-proximal-initialization.pdf` | `code-ready + evidence-ready` | `plan-pia` / `probe-pia-assets` / `dry-run-pia` / `runtime-probe-pia` / `run-pia-runtime-smoke` / `run-pia-synth-smoke` | [pia-runtime-smoke-cpu](../experiments/pia-runtime-smoke-cpu/summary.json), [pia-runtime-smoke-gpu](../experiments/pia-runtime-smoke-gpu/summary.json), [pia-synth-smoke-cpu](../experiments/pia-synth-smoke-cpu/summary.json), [pia-synth-smoke-gpu](../experiments/pia-synth-smoke-gpu/summary.json) | 缺真实 DDPM checkpoint 与数据集根目录 |

## 白盒

| 方法 | 论文 | 当前状态 | 仓库命令 | 当前证据 | 主要阻塞 |
| --- | --- | --- | --- | --- | --- |
| `white-box lead` | `2025-popets-white-box-membership-inference-diffusion-models.pdf` | `research-ready` | 暂无 | 暂无 | 缺可访问 checkpoint、训练配置、样本级梯度或激活接口 |
| `finding-nemo` | `2024-neurips-finding-nemo-localizing-memorization-neurons-diffusion-models.pdf` | `research-ready` | 暂无 | 暂无 | 缺 neuron-level 分析接口和相应模型资产 |

## 当前判断

1. 黑盒方向已经不是纯调研状态，至少有三条可执行证据链。
2. `recon` 已经有统一 mainline smoke，并且现在补上了 dataset/checkpoint 直达 score artifact 的 runtime mainline 入口；`Stable Diffusion + DDIM` 已经从 `1-sample` 提升到 `100-sample public runtime-mainline`，并且可以对比 `public-50` 与 `public-100` 两档指标变化，`kandinsky_v22` 的最小真实 runtime-mainline 与 artifact-mainline 也已经产出 summary。
3. `DiT` 官方采样路径不再只停留在 `2-step` 极小 smoke，当前已经有本地 checkpoint 驱动的 [dit-sample-step10](../experiments/dit-sample-step10/summary.json) 和 [dit-sample-step50](../experiments/dit-sample-step50/summary.json)，说明官方 `DiT-XL/2 256x256` 权重在本机可重复跑到 `50-step`。
4. 纯黑盒主线还没有真实 benchmark-ready 结果，因为 `recon` 当前仍是公开子集级证据，target/shadow/member/non-member 的论文级语义映射尚未最终核准；`DiT` 虽然已经有 `step10` 官方采样证据，但还没有进入与成员推断直接相连的实验协议，`variation` 仍停在 synthetic smoke、artifact summary、upstream smoke 或 dry-run 级别。
5. 下一步优先级建议：
   - 先核准 `recon` 公开资产的 target/shadow/member/non-member 映射，并解释 `public-50` 与 `public-100` 的指标差异
   - 再把 `Kandinsky` 扩到不止 `1-sample`，并视资源决定 `DiT` 是否继续往更高步数或更高分辨率推进
6. 研究仓库现在已经新增 Go 版本地 HTTP API 控制面，入口见 [local-api.md](/D:/Code/DiffAudit/Project/docs/local-api.md)。当前已打通 `health`、模型目录、`recon` 最佳证据查询、workspace summary 查询，以及 `recon_artifact_mainline` / `recon_runtime_mainline` 的受控 job 提交、列表与状态查询；真实实验仍继续由 Python CLI 执行，平台联调应优先对接这条 Go 控制面，而不是继续消费平台仓库内的 stub job 实现。

## 统一黑盒汇总

- [experiments/blackbox-status/summary.json](/D:/Code/DiffAudit/Project/experiments/blackbox-status/summary.json)

## 文档归档与飞书同步进度

更新时间：`2026-04-06 14:48:23 +08:00`
同步基线提交：`06c083d`

当前这部分状态描述的不是算法复现，而是论文整理、born-digital Markdown 精修、飞书三件套归档和总索引同步进度。

### 已完成批次

- 黑盒 `4 / 4` 已按新规范完成：
  - 报告
  - 精修原文
  - PDF
  - 链接可读权限
  - 总索引换链
- 灰盒首批 `4 / 4` 已按新规范完成：
  - `SecMI`
  - `Structural Memorization`
  - `CDI`
  - `MoFit`
- 灰盒第二批中，以下条目已完成飞书目录级归档：
  - `PIA`
  - `SiMA`
  - `Noise as a Probe`
  - `SIDE` 已完成报告、稳定版精修原文与 PDF，并已补回总索引展示字段
  - `small-noise-injection` 已完成报告、稳定版精修原文与 PDF，并已补回总索引展示字段

### 已确认可用的飞书三件套

- 黑盒：
  - `Towards Black-Box Membership Inference Attack for Diffusion Models`
  - `CLiD`
  - `Black-box Membership Inference Attacks against Fine-tuned Diffusion Models`
  - `Membership Inference Attacks for Face Images Against Fine-Tuned Latent Diffusion Models`
- 灰盒：
  - `SecMI`
  - `Structural Memorization`
  - `CDI`
  - `MoFit`
  - `PIA`
  - `SiMA`
  - `Noise as a Probe`
  - `SIDE`
  - `small-noise-injection`

### 当前规范状态

- 飞书总索引已改成团队展示口径：
  - 已刷新条目优先给出 `飞书 PDF`
  - 同时补 `GitHub PDF`
  - 补 `数据集与模型`
  - 补 `开源仓库`
  - 补 `阅读报告`
- PDF 链接策略已改为：
  - 优先引用飞书 PDF
  - 仅在飞书未上传或上传受限时退回 GitHub PDF
- 子代理职责已固定：
  - 只产出本地中文报告和精修原文
  - 不直接做飞书终态
  - 由主线程统一发布到飞书并换链

### 当前已知问题

- 一部分旧批次“精修原文”仍是偏短的笔记式版本，不是完整 born-digital 原文，需要继续逐篇替换。
- 某些精修原文在飞书同步时会因图片路径、caption 编码或长文分块方式触发告警；当前主线程已改为：
  - 先发稳定正文
  - 再补关键图
  - 必要时移除不稳定的 born-digital 内联碎图
- `SIDE` 精修原文已按上述策略重发完成，当前这类问题主要还会出现在其余旧批次条目上。
- 本地 `validate_report.py` 仍按旧模板校验，会对新规范文稿误报缺字段，尚未适配。

### 下一个 Agent 接手上下文

以下内容不是建议，而是当前机器上的真实接手上下文。后续 agent 应直接基于这里继续，不要重新摸索流程。

#### 1. 当前文档与归档主线

- 总索引飞书文档：
  - `https://www.feishu.cn/docx/ITzEdcyWSoXRqKxuLe3cx4yInEe`
- 飞书根目录：
  - `DiffAudit` folder token: `EpmAfeYfYlNNMgd9zwKc0xBCnZd`
- 论文总目录：
  - `论文` folder token: `FKbjfHlyMl3rCBdVEc5ciaCSnUA`
- 当前正式流程统一用：
  - `--as user`
- 本地飞书统一 CLI：
  - [feishu_ops.py](/D:/Code/DiffAudit/LocalOps/paper-pipeline/scripts/feishu_ops.py)

#### 2. 已知可直接用的论文目录 token

- 黑盒：
  - `Towards...` -> `L9lCfhhaTlFbiOduIBecfmLcnFb`
  - `CLiD` -> `PRDxf5KxQlaYlWd9Fzmca5ivnkf`
  - `NDSS black-box` -> `PqxffChZYl46ePdH9KMc4JTGnFf`
  - `VISAPP face` -> `GUl6fMTjMl9GxrdLemGcmkNVn0e`
- 灰盒：
  - `SecMI` -> `JEIofxdpklZoUHdxzGrcVi60n7g`
  - `SIDE` -> `A5HofCRBglXSMxdfiE9cAH8qnnh`
  - `Structural Memorization` -> `Y7ahfitvilSnBDdLNqUcNcHCnJb`
  - `PIA` -> `SkDMfYmH4lU8MAd1B5EcMa6mnDb`
  - `SiMA` -> `Rd7KfrUh9lVlFhdbEH8cZn53nEf`
  - `small-noise-injection` -> `Dhj2fm9zrloPoXdKLxUcTN3dnmf`
  - `CDI` -> `EecnfkAhbln1yidSyXOcl6mznic`
  - `Noise as a Probe` -> `AGcTfITlRlZj7zdpkIocM0bqn6f`
  - `MoFit` -> `ZjnkfJf1RlYIV6daD8Xc1GMrnAg`

#### 3. 当前最值得继续做的事

接手优先级按下面顺序：

1. 继续核对并把 `PIA / SiMA / Noise / SIDE / small-noise-injection` 的最终新链接与 PDF 链接完整回写到：
   - [references/materials/paper-index.md](/D:/Code/DiffAudit/Project/references/materials/paper-index.md)
   - [manifest.csv](/D:/Code/DiffAudit/Project/docs/paper-reports/manifest.csv)
   - [master-feishu-index.md](/D:/Code/DiffAudit/Project/docs/paper-reports/master-feishu-index.md)
2. 继续刷新尚未按新规范重发的其余灰盒文档。
3. 继续替换那些“只有短摘要”的旧精修原文。

#### 4. 当前需要避免重复踩的坑

- 不要让子代理直接做飞书终态。
  - 子代理只负责本地报告和精修原文。
  - 主线程统一做飞书发布、移动、权限和换链。
- 飞书正文更新不要直接用超长 PowerShell `$md` 命令串。
  - 超长文档会撞命令行长度限制。
  - 优先用 [feishu_ops.py](/D:/Code/DiffAudit/LocalOps/paper-pipeline/scripts/feishu_ops.py) 的 `sync-report` 子命令。
- PDF 链接策略已经改了：
  - 优先飞书 PDF
  - 飞书没有时才用 GitHub PDF
- 精修原文不要接受“只有一段摘要”的版本。
  - 若同步时图片不断报错，先发稳定正文，再补图。
- 两类常见飞书失败根因：
  - 中文标题通过 PowerShell 直接创建 doc 时的编码问题
  - 图片 caption 或 born-digital 相对路径导致 `media-insert` 失败

#### 5. 可直接复用的命令

创建或覆盖单篇报告：

```powershell
py -3 D:\Code\DiffAudit\LocalOps\paper-pipeline\scripts\feishu_ops.py sync-report `
  D:\Code\DiffAudit\Project\docs\paper-reports\<track>\<paper>-report.md `
  --doc "<existing-doc-url>" `
  --title "论文报告：<English Title>"
```

创建或覆盖单篇精修原文：

```powershell
py -3 D:\Code\DiffAudit\LocalOps\paper-pipeline\scripts\feishu_ops.py sync-report `
  D:\Code\DiffAudit\Project\docs\paper-reports\markdown\<track>\<paper>\<paper>-refined.md `
  --doc "<existing-doc-url>" `
  --title "OCR精修版：<English Title>"
```

上传 PDF：

```powershell
lark-cli drive +upload --as user `
  --file ".\references\materials\<track>\<paper>.pdf" `
  --name "<paper>.pdf" `
  --folder-token "<folder-token>"
```

移动 docx 到论文目录：

```powershell
lark-cli drive +move --as user `
  --file-token "<doc-token>" `
  --type docx `
  --folder-token "<folder-token>"
```

#### 6. 当前实际判断

- 文档线已经不是“从零搭流程”的阶段，而是“逐篇把旧批次替换成可展示、可交接、可复核版本”的阶段。
- 接手 agent 不应再重新设计模板或权限策略。
- 最短路径就是：
  - 选一篇旧条目
  - 检查本地 report/refined 是否已可用
  - 用现有脚本发布飞书三件套
  - 改 `paper-index.md / manifest.csv / master-feishu-index.md`
  - 再更新本文件的进度与剩余事项

### 下一步

1. 继续核对 `PIA / SiMA / Noise / SIDE / small-noise-injection` 在仓库侧索引文件中的最终链接回写是否全部一致。
2. 继续刷新尚未按新规范重发的其余灰盒文稿。
3. 逐篇替换仍然偏短的旧精修原文。
4. 继续把新飞书链接回写到：
   - `references/materials/paper-index.md`
   - `docs/paper-reports/manifest.csv`
   - `docs/paper-reports/master-feishu-index.md`
