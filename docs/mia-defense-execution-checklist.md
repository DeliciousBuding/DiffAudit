# MIA Defense Execution Checklist

这份清单基于 [mia-defense-document.docx](../references/materials/context/mia-defense-document.docx) 与当前仓库真实状态整理，目标是把“文档路线”变成“可执行任务”。

## 使用规则

- 这份清单只写当前仓库可执行或可判定的事项。
- `done` 表示仓库内已有明确证据。
- `pending` 表示路线成立，但还没有正式实现或正式结果。
- `blocked` 表示文档提出了方向，但当前资产或接口还不够。
- `not-adopted` 表示文档表述不适合直接进入当前主线。

## 0. 文档规范化

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| 将 `.docx` 纳入仓库 | `done` | [mia-defense-document.docx](../references/materials/context/mia-defense-document.docx) 已提交到 `main` |
| 生成可检索 Markdown | `done` | [mia-defense-document.md](../references/materials/context/mia-defense-document.md) |
| 生成正式研究索引 | `done` | [mia-defense-research-index.md](mia-defense-research-index.md) |
| 校正文档中的 threat-model / citation 口径 | `pending` | 已在研究索引标注问题，但尚未反写回原 `.docx` |

## 1. 文献与本地材料核对

| 文档中的关键文献 | 本地状态 | 下一步 |
| --- | --- | --- |
| `Shokri 2017` | `done` | 保留为 MIA 基础文献 |
| `Carlini First Principles 2022` | `done` | 后续统一口径到 `2022` |
| `SecMI` | `done` | 保持灰盒 baseline 身份 |
| `PIA` | `done` | 保持灰盒主线身份 |
| `TMIA-DM` | `done` | 已归档为灰盒候选论文，当前保持 research-ready |
| `Matsumoto 2023 SPW` | `done` | 作为扩散 MIA 综述补充 |
| `Dockhorn 2023` | `done` | 白盒 DP 防御参考 |
| `Ghalebikesabi 2023` | `done` | 白盒 DP 防御参考 |
| `MemGuard 2019` | `done` | 通用黑盒防御参考 |
| `Nasr 2018` | `done` | 通用训练期防御参考 |

## 2. 研究路线对齐

### 2.1 黑盒

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| 继续把 `recon` 作为当前 black-box 主线 | `done` | 当前仓库真实 black-box 主线不是 `SecMI` |
| 把 `SecMI` 从 black-box 执行清单里剥离 | `pending` | 需在团队共识文档和后续汇报里统一 |
| `B-1` 输出概率平滑设计成 black-box defense 原型 | `pending` | 当前没有正式实现 |
| `B-2` 查询限速设计成 black-box defense 原型 | `pending` | 当前没有正式实现 |
| 针对 `DDIM / SD1.5` 评估 black-box 防御成本 | `pending` | 还没有统一防御成本表 |

### 2.2 灰盒

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| 以 `PIA` 作为灰盒主线 | `done` | [manifest.json](../workspaces/gray-box/assets/pia/manifest.json) 和 mainline 已存在 |
| 将 `SecMI` 保持为灰盒 baseline | `done` | 代码与文献都已落地 |
| 将 `TMIA-DM` 纳入灰盒候选文献池 | `done` | 当前已归档并完成威胁模型判断，但还没有执行入口 |
| 把 `G-1` 映射为正式灰盒防御原型 | `done` | 当前已固定为 `provisional G-1 = stochastic-dropout` |
| 把 `G-2` 蒸馏代理模型纳入灰盒路线 | `blocked` | 当前没有正式蒸馏训练链和统一评估 |
| 把 `CFG 条件信号泄露` 变成独立实验项 | `pending` | 文档提出了问题，但仓库还没有正式实验入口 |

### 2.3 白盒

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| 以 `GSA` 作为白盒主线 | `done` | `GSA` 资产根、closed-loop 和 mainline 已落地 |
| 把 `W-1` 映射到 `DPDM / Diffusion-DP` 路线 | `done` | 白盒 defended comparator 已落到 [white-box table](../workspaces/white-box/2026-04-08-whitebox-attack-defense-table.md) |
| 设计 `步骤自适应梯度裁剪` 实验计划 | `pending` | 文档提出了方向，当前仓库未实现 |
| 把 `W-2` 成员信号对抗训练变成正式训练目标 | `blocked` | 当前还没有训练目标、评价口径和代码路径 |
| 把 `DDIM Inversion` 白盒泄露分析写成独立评估项 | `pending` | 有文献线索，缺统一实验入口 |

## 3. 资产与仓库检查

| 资产 / 仓库 | 状态 | 本地位置 | 下一步 |
| --- | --- | --- | --- |
| `PIA` 官方仓 | `done` | [external/PIA](../external/PIA) | 继续用于灰盒主线 |
| `SecMI` 官方仓 | `done` | [external/SecMI](../external/SecMI) | 继续用于 baseline / 资产探针 |
| `SecMI` adapter 子集 | `done` | [third_party/secmi](../third_party/secmi) | 保持最小集成 |
| `GSA` 官方仓 | `done` | [workspaces/white-box/external/GSA](../workspaces/white-box/external/GSA) | 继续用于白盒主线 |
| `DPDM` 仓 | `done` | [external/DPDM](../external/DPDM) | 纳入 `W-1` 路线 |
| `PIA` 真实资产根 | `done` | [workspaces/gray-box/assets/pia](../workspaces/gray-box/assets/pia) | 继续扩大样本和 provenance 核准 |
| `GSA` 真实资产根 | `done` | [workspaces/white-box/assets/gsa](../workspaces/white-box/assets/gsa) | 继续扩大 checkpoint 和 split 规模 |
| `CelebA-HQ` 扩展资产 | `pending` | 当前未见正式主线资产清单 | 只做 feasibility，不抢主线 |
| `ImageNet-64` 扩展资产 | `pending` | 当前未见正式主线资产清单 | 只做 feasibility，不抢主线 |

## 4. 统一评估框架待办

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| 统一 attack / defense 对比表模板 | `done` | 已新增 [unified table](../workspaces/implementation/2026-04-08-unified-attack-defense-table.md) |
| 统一记录 `AUC / ASR / TPR@低FPR / FID / IS / LPIPS / cost` | `pending` | 需要跨黑灰白统一字段 |
| 统一记录 defense 前后指标变化 | `done` | `PIA GPU512` 与 `W-1 strong-v3 full-scale` 已进入统一总表 |
| 统一记录 provenance / asset_grade / contract_stage | `done` | `PIA` 与 `GSA` 已进入 manifest 体系 |

## 5. 按当前仓库状态重排的执行顺序

### 第一优先级

- `PIA`：
  - 扩大样本规模
  - 核准 provenance
  - 把 `G-1` 变成正式灰盒 defense run
- `GSA`：
  - 扩大 bucket 数量
  - 提升训练 epoch
  - 形成更稳定的 closed-loop 指标
- `W-1 / DPDM`：
  - 把 [external/DPDM](../external/DPDM) 接入正式 white-box defense baseline

### 第二优先级

- black-box defense：
  - 把 `B-1` 输出平滑做成可插拔推理后处理
  - 把 `B-2` 查询限速做成平台 / Runtime 层策略原型

### 第三优先级

- `G-2` 知识蒸馏代理模型
- `W-2` 成员信号对抗训练
- `CelebA-HQ` / `ImageNet-64` 扩展验证

## 6. 当前最短路径结论

1. 这份文档最应该立刻服务的，不是重新定义攻击主线，而是补全 defense 设计与统一评估框架。
2. 当前仓库真实可落地的 defense 最短路径是：
   - 灰盒：`PIA + G-1`
   - 白盒：`GSA + W-1 (DPDM)`
   - 黑盒：`recon + B-1 / B-2`
3. 当前最不应该做的事，是把文档中不够严格的 threat-model 归类直接写回仓库状态文档或对外汇报。
4. `TMIA-DM` 当前已经证明“中文期刊里也有可用的灰盒时间序列信号路线”，但它不能被写成当前黑盒主线。

## 7. 申报阶段附加约束

这些约束来自 [和师兄聊天录音.txt](../../Archive/reference-materials/和师兄聊天录音.txt)，它们反映的是当前申报与答辩阶段的实务口径。

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| 优先保证“有可讲的攻击复现 + 一个能压指标的防御改动” | `pending` | 这是当前申报阶段最直接的成果要求 |
| 不把完整前端 / 产品平台作为当前第一优先级 | `done` | 录音明确认为静态展示先于完整网页 |
| `PIA` 论文必须自己读懂后再设计防御 | `pending` | 不能只跑代码或只靠 AI 摘要 |
| 申报 PPT 先讲背景、痛点、现有不足、我们的方法与预期效果 | `pending` | 当前需要和实验推进并行准备 |
| 不要求在申报阶段把所有系统与代码完全做完 | `done` | 更重要的是证明方向成立、工作量充足、路线可行 |
