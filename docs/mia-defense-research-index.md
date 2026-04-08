# MIA Defense Research Index

这份索引不是复述 [mia-defense-document.docx](../references/materials/context/mia-defense-document.docx)，而是把文档里的路线、文献、资产要求，映射到 DiffAudit 当前仓库的真实状态。

## 一句话判断

- 这份 `.docx` 是一份“研究策略文档”，不是“可直接照抄执行的复现规范”。
- 它最有价值的部分是：把扩散模型 MIA 的防御问题拆成黑盒、灰盒、白盒三条线，并明确写出了攻击面、候选防御、评估指标和任务分工。
- 它最需要纠偏的部分是：文内有少量威胁模型归类和文献年份/会议信息不够严格，不能直接当作当前仓库的真实执行口径。

## 当前应如何使用这份文档

建议把它用作三类输入：

1. `防御设计 backlog`
   - `B-1 / B-2 / W-1 / W-2 / G-1 / G-2` 这六条方法名，适合直接转成仓库内的 defense backlog。
2. `统一评估口径来源`
   - `ASR`、`AUC-ROC`、`TPR@低FPR`、`FID`、`IS`、`LPIPS`、训练/推理开销，这些指标口径适合进入统一表。
3. `文献和资产导航页`
   - 文档里点名的论文、数据集、模型、攻击主线，都可以映射到本地已有 PDF、代码仓库、manifest 和 runs。

不建议把它直接当作：

- 当前仓库的真实 threat-model 归类
- 当前仓库的已完成实验清单
- 当前项目的唯一主线执行计划

## 文档路线与当前仓库对照

| 维度 | 文档中的主张 | 当前仓库的真实映射 | 当前判断 |
| --- | --- | --- | --- |
| 黑盒攻击 | 文档任务分工把 `SecMI` 和 `Carlini 2021` 放进黑盒工作包 | 当前仓库黑盒主线是 `recon`，并保留 `CLiD`、`variation` 等路线；`SecMI` 在仓库中被归为灰盒 | 文档这里不能直接照搬 |
| 灰盒攻击 | 文档强调影子模型、迁移攻击、元分类器攻击 | 当前仓库灰盒主线是 `PIA`，`SecMI` 是 baseline；二者都有本地资产或代码骨架 | 文档和仓库大体一致，但需要把 `PIA/SecMI` 放在优先位 |
| 白盒攻击 | 文档把梯度、DDIM Inversion、中间激活差异作为白盒信号 | 当前仓库白盒主线是 `GSA`，另有 `Finding NeMo` 作为研究准备态 | 文档与仓库基本一致 |
| 黑盒防御 | `B-1` 输出概率平滑；`B-2` 查询限速 | 当前仓库没有正式 black-box defense mainline | 适合作为新 backlog，不是现成能力 |
| 白盒防御 | `W-1` Diffusion-DP；`W-2` 成员信号对抗训练 | 当前仓库已有 [external/DPDM](../external/DPDM)；`W-2` 还没有正式训练目标与实现 | `W-1` 可落地，`W-2` 需要先定义 |
| 灰盒防御 | `G-1` 推理时架构随机化；`G-2` 蒸馏代理模型 | 当前仓库已有 `PIA` 的 dropout-defense 原型证据，但没有形成正式灰盒 defense 规范 | `G-1` 有近似原型，`G-2` 基本未开始 |
| 统一评估 | 文档要求三盒统一指标、统一表格、统一成本评估 | 当前仓库已有状态文档和 method-specific summaries，但还没有把所有 attack/defense 合到单一对照表 | 文档这部分值得直接吸收 |

## 文档里的关键纠偏点

### 1. `SecMI` 不应直接当作当前 black-box 主线

文档在任务分工中把 `SecMI` 放进黑盒工作包，但当前仓库和主流理解都把 `SecMI` 当作 diffusion MIA 的灰盒 / partial-observability 基线。

本地证据：

- [2023-icml-secmi-membership-inference-diffusion-models.pdf](../references/materials/gray-box/2023-icml-secmi-membership-inference-diffusion-models.pdf)
- [third_party/secmi](../third_party/secmi)
- [configs/attacks/secmi_plan.yaml](../configs/attacks/secmi_plan.yaml)
- [docs/reproduction-status.md](reproduction-status.md)

### 2. 文档里的 `SecMI（NeurIPS 2023）` 口径需要统一

当前本地 canonical 文件和项目文档都按 `ICML 2023` / `PMLR` 处理：

- [2023-icml-secmi-membership-inference-diffusion-models.pdf](../references/materials/gray-box/2023-icml-secmi-membership-inference-diffusion-models.pdf)
- [manifest.csv](../references/materials/manifest.csv)

### 3. `Carlini 2021` 与 `First Principles 2022` 不能混写

这份 `.docx` 在“攻击基准”里写 `Carlini 2021`，但在关键参考文献里实际列的是 `Membership Inference Attacks From First Principles. IEEE S&P 2022`。

当前仓库已有本地 canonical 文件：

- [2022-ieee-membership-inference-first-principles.pdf](../references/materials/survey/2022-ieee-membership-inference-first-principles.pdf)

后续任何执行清单、汇报或表格都应统一到明确标题和年份。

### 4. 文档是“统一防御框架草案”，不是当前主线进度报告

它没有对本地仓库当前已经完成的：

- `PIA real-asset runtime mainline`
- `GSA real-asset closed loop ready`
- `recon` public runtime mainline

做同步，所以不能直接拿它来判断“当前完成了多少”。

## 文献索引与本地落点

| 文档角色 | 论文 / 文献 | 本地文件 | 本地阅读或工程落点 | 当前状态 |
| --- | --- | --- | --- | --- |
| MIA 基础 | Carlini et al. First Principles | [2022-ieee-membership-inference-first-principles.pdf](../references/materials/survey/2022-ieee-membership-inference-first-principles.pdf) | [paper-index.md](../references/materials/paper-index.md) | 已落地 |
| MIA 基础 | Shokri et al. 2017 | [2017-ieee-membership-inference-machine-learning-models.pdf](../references/materials/survey/2017-ieee-membership-inference-machine-learning-models.pdf) | [paper-index.md](../references/materials/paper-index.md) | 已落地 |
| 灰盒基线 | SecMI | [2023-icml-secmi-membership-inference-diffusion-models.pdf](../references/materials/gray-box/2023-icml-secmi-membership-inference-diffusion-models.pdf) | [third_party/secmi](../third_party/secmi) | 代码已接入，真实资产仍待强化 |
| 灰盒主线 | PIA | [2024-iclr-pia-proximal-initialization.pdf](../references/materials/gray-box/2024-iclr-pia-proximal-initialization.pdf) | [external/PIA](../external/PIA), [assets/pia/manifest.json](../workspaces/gray-box/assets/pia/manifest.json) | 已有 real-asset mainline |
| 灰盒候选 | TMIA-DM | [2026-crad-temporal-membership-inference-attack-method-diffusion-models.pdf](../references/materials/gray-box/2026-crad-temporal-membership-inference-attack-method-diffusion-models.pdf) | [2026-04-09-tmia-dm-intake.md](../workspaces/gray-box/2026-04-09-tmia-dm-intake.md) | 已归档，当前仅到 research-ready |
| 白盒主线 | GSA | [2025-popets-white-box-membership-inference-diffusion-models.pdf](../references/materials/white-box/2025-popets-white-box-membership-inference-diffusion-models.pdf) | [workspaces/white-box/external/GSA](../workspaces/white-box/external/GSA), [cifar10-ddpm-mainline.json](../workspaces/white-box/assets/gsa/manifests/cifar10-ddpm-mainline.json) | 已有 real-asset closed loop |
| 白盒扩展 | Finding NeMo | [2024-neurips-finding-nemo-localizing-memorization-neurons-diffusion-models.pdf](../references/materials/white-box/2024-neurips-finding-nemo-localizing-memorization-neurons-diffusion-models.pdf) | [paper-index.md](../references/materials/paper-index.md) | 研究准备态 |
| DP 防御 | Dockhorn et al. | [2023-tmlr-differentially-private-diffusion-models.pdf](../references/materials/survey/2023-tmlr-differentially-private-diffusion-models.pdf) | [external/DPDM](../external/DPDM) | 有仓库，未接成主线 |
| DP 防御 | Ghalebikesabi et al. | [2023-arxiv-differentially-private-diffusion-models-generate-useful-synthetic-images.pdf](../references/materials/survey/2023-arxiv-differentially-private-diffusion-models-generate-useful-synthetic-images.pdf) | [external/DPDM](../external/DPDM) | 有文献与仓库 |
| 通用防御 | MemGuard | [2019-ccs-memguard-defending-black-box-membership-inference.pdf](../references/materials/survey/2019-ccs-memguard-defending-black-box-membership-inference.pdf) | 文献级参考 | 仅文献 |
| 通用防御 | Adversarial Regularization | [2018-ccs-membership-privacy-adversarial-regularization.pdf](../references/materials/survey/2018-ccs-membership-privacy-adversarial-regularization.pdf) | 文献级参考 | 仅文献 |

## 本地资产索引

| 资产类型 | 本地位置 | 作用 | 当前判断 |
| --- | --- | --- | --- |
| 文档原件 | [mia-defense-document.docx](../references/materials/context/mia-defense-document.docx) | 内部研究策略源文件 | 已纳入仓库 |
| 文档检索版 | [mia-defense-document.md](../references/materials/context/mia-defense-document.md) | 便于搜索、比对和引用 | 本次新增 |
| PIA 代码仓 | [external/PIA](../external/PIA) | 灰盒主线实现来源 | 已就位 |
| SecMI 代码仓 | [external/SecMI](../external/SecMI) 与 [third_party/secmi](../third_party/secmi) | 灰盒 baseline 与 adapter 子集 | 已就位 |
| GSA 代码仓 | [workspaces/white-box/external/GSA](../workspaces/white-box/external/GSA) | 白盒主线实现来源 | 已就位 |
| DPDM 代码仓 | [external/DPDM](../external/DPDM) | White-box `W-1` / DP 防御候选 | 已就位但未纳主线 |
| PIA 资产根 | [workspaces/gray-box/assets/pia](../workspaces/gray-box/assets/pia) | checkpoint、dataset、provenance | 已形成规范资产根 |
| GSA 资产根 | [workspaces/white-box/assets/gsa](../workspaces/white-box/assets/gsa) | bucket、checkpoint-*、manifest | 已形成规范资产根 |
| 黑盒重建攻击仓 | [external/Reconstruction-based-Attack](../external/Reconstruction-based-Attack) | 当前 black-box 主线仓库 | 已就位 |
| 黑盒 CLiD 仓 | [external/CLiD](../external/CLiD) | 黑盒/近灰盒条件差异路线参考 | 已就位 |

## 基于这份文档的正式结论

### 应吸收的内容

- 三种威胁模型拆分
- 三类扩散特有攻击面
- 六个 defense method id：`B-1`、`B-2`、`W-1`、`W-2`、`G-1`、`G-2`
- 统一评估口径：隐私指标、质量指标、成本指标、自适应攻击

### 不能直接照抄的内容

- `SecMI` 作为 black-box 工作包
- `Carlini 2021` 的模糊表述
- `SecMI NeurIPS 2023` 的会议信息
- 文档中的任务分工表，因为它没有映射到当前仓库真实资产与当前进度

### 当前最合理的用法

把这份 `.docx` 当成：

- `防御方向总纲`
- `统一评估指标来源`
- `文献与资产补齐导航`

不要把它当成：

- `当前仓库真实进度表`
- `当前 attack taxonomy 的最终权威口径`

## 与当前主线的最短对接方式

1. 黑盒继续以现有 `recon` 主线为执行主线，把文档中的 `B-1 / B-2` 视为防御 backlog，而不是把 `SecMI` 生搬进 black-box。
2. 灰盒直接用文档推进 `PIA + SecMI + G-1/G-2` 的防御设计，其中 `PIA` 已经具备最好的仓库接入基础。
3. `TMIA-DM` 已归档后，应被视为 `PIA` 同轴的灰盒候选论文，而不是黑盒主线替代。
4. 白盒直接把文档中的 `W-1 / W-2` 映射到 `GSA + DPDM` 路线，优先做 `W-1`，因为本地已经有 [external/DPDM](../external/DPDM)。
5. 所有后续汇报都应显式区分：
   - `文档提出的研究路线`
   - `仓库当前已完成的真实状态`

## 从师兄聊天录音补出的执行约束

来源文件：

- [和师兄聊天录音.txt](../../Archive/reference-materials/和师兄聊天录音.txt)

录音补出的约束，不是算法真理，而是当前申报阶段的执行口径：

1. 当前阶段优先级不是“完整产品做完”，而是“先有可讲的攻击复现结果，再加一个能把指标打下来的防御改动”。
2. 前端 / 网页端不是当前主任务。录音明确认为申报阶段只需要静态展示材料，不需要先做完整交互平台。
3. 黑盒被视为最容易先出结果的一条线，但录音同时点名 `PIA` 论文必须自己读懂，说明灰盒也被视为重要抓手。
4. 防御的思路不是抽象地“给用户建议”，而是要找出攻击依赖的可分信号，再把这个依赖关系切断，看 `AUC / ASR / TPR` 是否下降。
5. 申报阶段不要求把所有代码和系统完全做完，更强调：
   - 背景讲清楚
   - 痛点讲清楚
   - 现有方法不足讲清楚
   - 我们的路线与预期效果讲清楚
6. 这意味着当前仓库里的正式研究索引和执行清单，应同时服务两个目标：
   - 真实实验推进
   - 申报 / PPT 叙事准备
