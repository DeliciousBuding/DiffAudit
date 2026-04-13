# 2026-04-09 Claude 4.9 报告复核与后续研究判断

这份文档用于复核 `D:\Code\DiffAudit\docs\ClaudeReport\4.9.md` 中提到的新增论文与路线建议，并把它们放回 `Research` 当前真实主线中理解。

它不是新的路线图，也不替代：

- `docs/comprehensive-progress.md`
- `docs/reproduction-status.md`
- `ROADMAP.md`

当前唯一目的只有两个：

1. 校正 `4.9.md` 中可以直接引用、需要降级表述、或需要补证据的部分
2. 说明这些新论文对 DiffAudit 当前阶段到底应该怎样参考

## 1. 当前主线约束

截至 `2026-04-09`，`Research` 当前真实主线仍然是：

- 黑盒风险证据：`recon DDIM public-100 step30`
- 灰盒主讲线：`PIA + provisional G-1(all_steps)`
- 白盒深度线：`GSA rerun1 + W-1 strong-v3 full-scale`
- 当前唯一 active GPU 问题：`white-box same-protocol bridge`

这意味着：

- 任何新论文都不能直接改写当前主讲线
- 任何新论文都不能抢走 `PIA provenance` 收口与白盒 bridge 决策的优先级
- 新文献的首要价值应是：
  - 帮当前结果建立更稳的解释框架
  - 帮下一阶段问题做候选排序
  - 帮答辩或申报叙事补边界条件，而不是立刻扩新 GPU 任务

## 2. 对 `4.9.md` 的总体判断

`4.9.md` 的方向判断大体有价值，但当前版本不能直接当项目级结论引用，主要有四个问题。

第一，文档正文从中段开始重复了一遍，同一批结论在后半段再次出现，当前应先去重。

第二，部分材料名称与论文标题混用。最典型的是 `CopyMark`：当前更稳妥的写法应是“论文标题是 `Real-World Benchmarks Make Membership Inference Attacks Fail on Diffusion Models`，`CopyMark` 是论文提出的 benchmark 名称”，而不是把 `CopyMark` 当作正式论文标题。

第三，部分 venue 口径需要降级到“当前可核验来源”。例如：

- `Privacy-Preserving Low-Rank Adaptation against Membership Inference Attacks for Latent Diffusion Models` 本地索引已按 `AAAI 2025` 记录，不应在没有二次核验前写成 `ICCV 2025`
- `On the Edge of Memorization in Diffusion Models` 当前已核验到 arXiv 与 OpenReview preprint，但不应直接写成确定的 `NeurIPS 2025`

第四，`4.9.md` 默认把“有价值的新论文”直接推成“应尽快进入执行面”，这与当前研究阶段不一致。当前真正允许进入执行层的新问题，仍然要服从 `Phase D -> Phase E` 的 gate。

## 3. 逐篇复核与分级

### 3.1 频域视角论文：可以立即参考，但先作为解释层而不是新主线

- 论文：`Unveiling Impact of Frequency Components on Membership Inference Attacks for Diffusion Models`
- 当前可核验来源：`arXiv:2505.20955`
- 本地 PDF：`references/materials/survey/2025-arxiv-enhancing-membership-inference-attacks-frequency-domain-perspective.pdf`

这篇论文最有价值的地方，不是它提出了一个全新攻击家族，而是它把现有扩散模型 MIA 统一到 membership score 范式下，并指出高频信息处理缺陷会系统性干扰成员判别。它与 DiffAudit 当前三条攻击线的关系是“解释增强”而不是“路线替代”。

对当前项目最合适的参考方式是做一层 post-hoc analysis：

- 对 `recon` 结果补高频/低频分组统计
- 对 `PIA` 的 `epsilon-trajectory consistency` 信号做频率分解
- 对 `GSA` 的 member / non-member separability 检查是否也受频率偏置影响

这类工作主要消耗分析时间和少量复算，不要求马上开新的主 GPU 任务。因此，这篇论文适合立刻进入“当前结果解释层”，也是 `4.9.md` 里最值得先吸收的一篇。

### 3.2 CopyMark 论文：应立刻进入答辩叙事，但不应被误读为“现有结果无效”

- 论文：`Real-World Benchmarks Make Membership Inference Attacks Fail on Diffusion Models`
- 论文中提出的 benchmark：`CopyMark`
- 当前可核验来源：`OpenReview forum id = EE2tIwKhSW`
- 本地 PDF：`references/materials/survey/2024-arxiv-real-world-benchmarks-membership-inference-attacks-fail-diffusion-models.pdf`

这篇论文的真正价值，是逼我们把“攻击有效的边界条件”说清楚。它并不是在否定所有扩散模型 MIA，而是在指出：如果评测面切到更真实的预训练模型、去掉分布偏置、统一 pipeline，既有 MIA 的效果会明显下降。

对 DiffAudit 来说，这篇论文最适合放在两个地方：

- 黑盒叙事的有效性边界说明
- `PIA / recon / GSA` 结果的外推边界说明

但当前不应据此得出“本地主线结果被推翻”的结论。原因很简单：DiffAudit 当前很多 admitted 结果本来就是在明确、受控、偏研究型的协议面上建立的，而不是在“预训练大模型现实版权取证”设定下建立的。更合理的写法应该是：

- 当前结果证明“在给定协议与资产条件下，成员信号确实存在且可被利用”
- `CopyMark` 提醒我们“这类信号是否能迁移到更真实设定，必须单独验证”

因此，这篇论文适合立即纳入项目叙事，但其作用是“加边界”，不是“重置主线”。

### 3.3 `Finding NeMo`、`Exploring Local Memorization`、`FB-Mem`：白盒扩展线成立，但不该抢当前 bridge 优先级

- `Finding NeMo: Localizing Neurons Responsible for Memorization in Diffusion Models`
- `Exploring Local Memorization in Diffusion Models via Bright Ending Attention`
- `Demystifying Foreground-Background Memorization in Diffusion Models`

本地已有 `Finding NeMo` 原文与报告；另外两篇现在也已经补到 `survey/` 目录。

这组三篇论文共同指向一个判断：如果白盒研究只停在“成员能不能被分出来”，那还不够；下一步自然会走向：

- 记忆到底集中在哪些内部单元
- 记忆是否是局部的，而不是整图级的
- 现有剪枝/神经元停用式缓解，为什么无法真正消掉局部记忆

这对 DiffAudit 的价值非常清楚，但它们当前更像 `Phase E` 的白盒扩展候选，而不是 `Phase D` 的直接任务。当前 `Phase D` 的问题仍然是：

- `GSA rerun1` 与 `W-1 strong-v3 full-scale` 的 same-protocol bridge 到底如何收口

所以更合理的顺序是：

1. 先把白盒 bridge 的合同和失败模式写清
2. 再决定是否把 `Finding NeMo + local memorization` 提升成下一阶段的机制线

换句话说，`4.9.md` 把这个方向看成“白盒扩展线”是对的，但如果现在就把注意力切到 neuron localization 或 local memorization mitigation，会让当前主任务失焦。

### 3.4 `DP-LoRA`：值得保留，但当前只能是 `W-1` 之后的替代防御候选

- 论文：`Privacy-Preserving Low-Rank Adaptation against Membership Inference Attacks for Latent Diffusion Models`
- 当前可核验来源：`arXiv:2402.11989`，`DBLP/AAAI 2025`
- 本地 PDF：`references/materials/survey/2025-aaai-privacy-preserving-lora-membership-inference-latent-diffusion-models.pdf`

`DP-LoRA` 的价值是把“差分隐私防御”从全模型 DP-SGD，收缩到 LoRA 级别的参数高效适配。这对工程可用性显然更友好，也比继续堆更重的 `DPDM` 训练更贴近真实部署。

但它当前不应替换 `W-1 = DPDM`，原因有三个：

- DiffAudit 当前白盒防御读链已经围绕 `DPDM/W-1` 建好，强行换题会破坏当前 admitted 结构
- `DP-LoRA` 对应的攻击面和模型条件与当前 `GSA + DDPM + CIFAR-10` 协议并不完全同构
- 当前白盒真正未解决的问题不是“没有替代防御候选”，而是“same-protocol bridge 该怎么收口”

因此，`DP-LoRA` 当前最合适的定位是：

- `W-1` 之后的下一代轻量防御候选
- 可在 `Phase E` 立项时作为 `W-2` 或独立 white-box defense candidate
- 暂不进入当前 admitted 主表

### 3.5 `On the Edge of Memorization`：理论价值高，但离当前执行面最远

- 论文：`On the Edge of Memorization in Diffusion Models`
- 当前可核验来源：`arXiv:2508.17689`
- 本地 PDF：`references/materials/survey/2025-arxiv-on-the-edge-of-memorization-diffusion-models.pdf`

这篇论文的价值确实在于提供“何时从泛化进入记忆”的理论视角，尤其是 phase transition / crossover point 的说法，很适合拿来解释为什么某些协议面下攻击信号会更强。

但它当前离 DiffAudit 的直接执行面最远，原因是：

- 当前项目的 admitted 结果主要来自具体协议与资产条件，而不是参数化理论实验室
- 这篇论文更适合作为解释框架或未来理论补强，而不是当前 runner / comparator / intake 层的直接输入

所以它应保留为“理论支撑材料”，但不建议在当前阶段投入实现精力。

### 3.6 `MIDST Challenge`：提醒“跨域不可直接外推”，但不应反向主导图像主线

- 论文：`MIDST Challenge at SaTML 2025: Membership Inference over Diffusion-models-based Synthetic Tabular data`
- 当前可核验来源：`arXiv:2603.19185`
- 本地 PDF：`references/materials/survey/2026-arxiv-midst-challenge-membership-inference-diffusion-models-synthetic-tabular-data.pdf`

`MIDST` 的价值不是告诉我们“图像域方法应该照搬到表格域”，恰恰相反，它说明跨域时攻击信号和有效方法会明显变化。这个结论对 DiffAudit 很有帮助，因为它提醒我们：

- 当前项目可以讲“分层审计框架”
- 但不应过度宣称“跨域通用攻击框架已经成立”

因此，`MIDST` 应该进入的是“边界声明”和“外推风险”部分，而不是当前算法主线。

## 4. 当前建议采用的引用口径

如果要把 `4.9.md` 中的新增文献真正写回项目叙事，当前建议采用下面这套分级。

### 4.1 可以立即进入当前项目叙事

- `Real-World Benchmarks Make Membership Inference Attacks Fail on Diffusion Models / CopyMark`
  - 用来约束外推边界
  - 用来解释为什么当前结果不能直接讲成“真实预训练模型版权取证已经成立”
- `Unveiling Impact of Frequency Components on Membership Inference Attacks for Diffusion Models`
  - 用来扩展现有 admitted 结果的解释层
  - 可以低成本进入 post-hoc analysis

### 4.2 进入 `Phase E` 候选池，但当前不抢主线

- `Finding NeMo`
- `Exploring Local Memorization in Diffusion Models via Bright Ending Attention`
- `Demystifying Foreground-Background Memorization in Diffusion Models`
- `Privacy-Preserving Low-Rank Adaptation against Membership Inference Attacks for Latent Diffusion Models`

这些论文共同构成下一阶段最自然的两条分支：

- 白盒机制定位与局部记忆线
- 轻量级差分隐私防御替代线

### 4.3 保留为理论或跨域边界材料

- `On the Edge of Memorization in Diffusion Models`
- `MIDST Challenge at SaTML 2025`

它们有价值，但不应主导当前执行顺序。

## 5. 对 `4.9.md` 中三个新增问题的重排

`4.9.md` 提了三个新增问题：频域感知、`DPDM -> DP-LoRA`、真实预训练模型评估。当前更合理的优先级应该重排为：

1. 频域感知的成员信号复核
2. 真实设定下的有效性边界叙事补强
3. `DP-LoRA` 作为下一阶段防御候选

原因是：

- 第一项最便宜，最不打断主线
- 第二项对答辩叙事最重要，但真正执行成本很高，因此先写边界，再决定是否开题
- 第三项是新防御线，不应在 `W-1` 未收口时提前切换

## 6. 当前建议执行顺序

如果要把这轮深研真的转成下一步动作，当前建议顺序如下。

### 6.1 现在就做

- 保持 `PIA` 主讲线与 `checkpoint/source provenance` blocker 口径不变
- 保持白盒 active 问题仍然是 same-protocol bridge，不切题
- 在现有 admitted 结果上增加一轮 frequency-aware 分析设计
- 在主线叙事或答辩材料里补上 `CopyMark` 式现实边界说明

### 6.2 当前只建 intake，不启动执行

- `Finding NeMo + local memorization + FB-Mem`
- `DP-LoRA`

### 6.3 当前不建议直接推进

- 在 `SD1.5 / SDXL` 预训练模型上立刻重做整套 `recon / PIA / GSA`
- 把表格域 `MIDST` 的 learned signal 直接外推成图像域结论
- 把 `On the Edge` 理论框架直接写成当前 admitted 结果的定量解释

## 7. 本次补齐的原文

本次已补下载以下 PDF 到 `Research` 仓库：

- `references/materials/survey/2025-arxiv-enhancing-membership-inference-attacks-frequency-domain-perspective.pdf`
- `references/materials/survey/2024-arxiv-real-world-benchmarks-membership-inference-attacks-fail-diffusion-models.pdf`
- `references/materials/survey/2025-iclr-exploring-local-memorization-diffusion-models-bright-ending-attention.pdf`
- `references/materials/survey/2025-arxiv-demystifying-foreground-background-memorization-diffusion-models.pdf`
- `references/materials/survey/2025-arxiv-on-the-edge-of-memorization-diffusion-models.pdf`
- `references/materials/survey/2026-arxiv-midst-challenge-membership-inference-diffusion-models-synthetic-tabular-data.pdf`

## 8. 当前结论

`4.9.md` 的方向判断不是问题，真正的问题是它没有把“当前主线约束”和“新文献价值”分层。

对 DiffAudit 来说，这轮新增文献里最应该立刻吸收的，不是新的大规模执行题，而是两件更克制的事：

- 用 `CopyMark` 校正对外叙事的边界
- 用频域论文增强对现有 admitted 结果的解释

其余像 `Finding NeMo` 的局部化扩展、`FB-Mem` 的局部记忆缓解、`DP-LoRA` 的轻量防御替代、`On the Edge` 的理论框架，都有价值，但更适合进入当前主线收口之后的 `Phase E` 候选池，而不是现在直接抢占执行面。

## 9. 外部核验来源

- 频域论文：`https://arxiv.org/abs/2505.20955`
- `CopyMark / Real-World Benchmarks Make Membership Inference Attacks Fail on Diffusion Models`：`https://openreview.net/forum?id=EE2tIwKhSW`
- `Exploring Local Memorization in Diffusion Models via Bright Ending Attention`：`https://arxiv.org/abs/2410.21665`
- `Demystifying Foreground-Background Memorization in Diffusion Models`：`https://arxiv.org/abs/2508.12148`
- `On the Edge of Memorization in Diffusion Models`：`https://arxiv.org/abs/2508.17689`
- `MIDST Challenge at SaTML 2025`：`https://arxiv.org/abs/2603.19185`
- `Privacy-Preserving Low-Rank Adaptation against Membership Inference Attacks for Latent Diffusion Models`：`https://dblp.org/rec/conf/aaai/GaoWZM25.html`

## 10. 落地承接文档（不改写主线）

为避免把“文献价值”直接误读成“新增执行线”，本仓库把两件当前最应吸收的内容单独落在以下承接文档中：

- 黑盒 `recon` 解释层（`CopyMark` 现实边界 + 频域 post-hoc 设计）：[workspaces/black-box/2026-04-10-recon-explanation-layer.md](../workspaces/black-box/2026-04-10-recon-explanation-layer.md)
- `Phase E` intake 固定队列（进入/退出条件与预期产物）：[future-phase-e-intake.md](future-phase-e-intake.md)
