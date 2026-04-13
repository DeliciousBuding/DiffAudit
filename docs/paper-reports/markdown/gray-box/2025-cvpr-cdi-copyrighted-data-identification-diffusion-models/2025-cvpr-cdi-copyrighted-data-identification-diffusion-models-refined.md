# 扩散模型中的版权数据识别
CDI: Copyrighted Data Identification in Diffusion Models

## 文档说明

- GitHub PDF：[2025-cvpr-cdi-copyrighted-data-identification-diffusion-models.pdf](https://github.com/DeliciousBuding/DiffAudit-Research/blob/main/references/materials/gray-box/2025-cvpr-cdi-copyrighted-data-identification-diffusion-models.pdf)
- 对应报告：[论文报告：CDI: Copyrighted Data Identification in Diffusion Models](https://www.feishu.cn/docx/QRzhdNv6NoryLIxPbz7crXbRnd5)
- 对应展示稿：[2025-cvpr-cdi-copyrighted-data-identification-diffusion-models-report.md](https://github.com/DeliciousBuding/DiffAudit-Research/blob/main/docs/paper-reports/gray-box/2025-cvpr-cdi-copyrighted-data-identification-diffusion-models-report.md)
- 开源实现：[sprintml/copyrighted_data_identification](https://github.com/sprintml/copyrighted_data_identification)
- 整理说明：本稿基于 born-digital Markdown 精修，只保留论文主线、关键公式、核心结果与对 DiffAudit 直接相关的结论，便于后续检索与引用。

---

# CDI: Copyrighted Data Identification in Diffusion Models

Jan Dubinski, Antoni Kowalczuk, Franziska Boenisch, Adam Dziedzic

## 摘要精修

论文讨论的是扩散模型训练数据的版权识别问题，但目标不是判断“某一张图像是不是训练成员”，而是判断“一个数据拥有者的一组作品是否整体参与了模型训练”。作者首先重新评估现有扩散模型成员推断，指出当模型和训练集规模增大时，单样本信号通常不够稳定，难以直接用于版权主张。为此，论文提出 Copyrighted Data Identification (CDI)，把现有 MIA 与新增特征提取为样本级证据，经评分模型选择性聚合后，再通过统计检验输出集合级结论。实验显示，CDI 在多类扩散模型上都能工作，在最有利的 COCO 文本条件模型上，只需约 `70` 个样本即可在超过 `99%` 置信度下识别训练使用。

## 1. 引言

论文的出发点很明确：当扩散模型不会在推理时逐字逐像素复现训练图像时，数据拥有者仍然需要一种方法证明自己的作品被模型使用。作者指出，成员推断攻击在概念上似乎适合这个任务，但现实中的大规模扩散模型训练成本很高，影子模型路线几乎不可行，而且现有单样本 MIA 在大模型上也缺乏足够稳定的区分能力。

基于这一观察，论文把视角从单样本成员推断转向 dataset inference。作者认为，现实中的版权争议通常涉及的是一组作品，而不是单个样本。只要能够把多张作品中的弱成员信号选择性聚合，并辅以统计显著性检验，就有机会得到比单样本更稳健的审计证据。

## 2. 背景与问题设定

论文首先回顾扩散模型的基础训练目标。对潜空间扩散模型，作者使用

$$
\mathcal{L}(z,t,\epsilon;f_{\theta})=\left\lVert \epsilon-f_{\theta}(z_t,t)\right\rVert_2^2
$$

作为最基础的噪声预测损失。已有扩散模型 MIA 大多围绕这个损失或其相邻扩散步差异构造成员性信号，例如 Denoising Loss、SecMIstat、PIA、PIAN。

CDI 的威胁模型是由可信第三方仲裁者执行的审计流程。仲裁者从受害方拿到公开嫌疑集合 `P` 和同分布未公开集合 `U`，再对目标模型执行灰盒或白盒审计。灰盒访问只要求在任意扩散步 `t` 上查询噪声预测；白盒访问还允许读取内部梯度和参数。论文特别强调，方法结论只适用于拥有 `P`、`U` 与相应模型访问权限的场景。

## 3. 单样本 MIA 的局限

论文在提出 CDI 之前，先系统评估已有扩散模型 MIA 在大规模模型上的能力。作者的结论是：单样本成员推断确实存在一定信号，但在训练集规模较大、模型较强时，这个信号并不足以支持高置信度版权主张。换言之，单样本分数可以作为线索，却很难单独成为证据。

这一步在论文中的作用很重要。CDI 不是“直接跳到数据集级”，而是在先确认单样本路线不足之后，再说明为何必须引入集合级判定。也正因为如此，CDI 不是对旧攻击的简单平均，而是试图建立更完整的证据链。

## 4. CDI 方法

### 4.1 特征提取

CDI 先利用现有扩散模型 MIA 提取基础成员性特征，然后引入三种新增特征：`Gradient Masking`、`Multiple Loss` 与 `Noise Optimization`。其中 `Multiple Loss` 最直接，它在多个扩散步同时计算损失，让评分器从时间维度选择更有辨识度的信号。

`Gradient Masking` 是论文最关键的新增特征。作者先计算

$$
g=\left|\nabla_{z_t}\mathcal{L}(z,t,\epsilon;f_{\theta})\right|
$$

然后取梯度幅值前 `20%` 的位置构成掩码 `M`，再用噪声替换这些位置得到

$$
\hat{z}_t=\epsilon\cdot M+z_t\cdot \neg M.
$$

直觉上，这一步会破坏对损失最重要的潜表示区域，再考察模型恢复这些语义区域的能力。论文报告该特征对 CDI 的提升最大，但它依赖梯度，因此更偏白盒访问。

`Noise Optimization` 则把监督学习里“成员样本更难被扰动改变预测”的观察迁移到扩散模型。作者在 `t=100` 上对带噪潜变量施加无界扰动，用 5 步 L-BFGS 优化最小化噪声预测损失，并把优化后的损失与扰动幅度当作新特征。

### 4.2 评分模型与统计检验

CDI 的核心不是某个单一特征，而是特征如何被组合。作者把 `P` 和 `U` 分成控制集与测试集，使用 `P_ctrl`、`U_ctrl` 上的特征训练一个逻辑回归评分模型 `s`，再把它应用到 `P_test`、`U_test` 上，得到每个样本的成员性分数。

随后，论文不直接对这些分数设阈值，而是进行集合级假设检验：

$$
H_0:\ \overline{s(fe(P_{\mathrm{test}}))}\le \overline{s(fe(U_{\mathrm{test}}))}.
$$

作者使用单尾 Welch `t` 检验判断是否拒绝 `H_0`，并通过 5 折交叉验证最大化样本利用率，再重复 1000 次随机采样并聚合 `p` 值，降低一次划分的偶然性。

![](_page_1_Figure_0.jpeg)

上图展示了 CDI 的完整流程：准备嫌疑集合 `P` 与对照集合 `U`，提取成员性特征，训练评分模型，再以统计检验得出集合级训练相关性结论。这张图准确体现了论文的方法边界，即它依赖的是“整组数据的弱信号聚合”，而不是某个单样本的强记忆。

## 5. 实验与结果

论文在 LDM、U-ViT、DiT 三类扩散模型上评估 CDI，覆盖无条件、类条件、文本条件、多分辨率以及 ImageNet 与 COCO 两类训练数据。实验中始终令 `|P|=|U|`，并通过 5 折交叉验证生成测试分数。除了主结果，论文还评估了统计检验是否必要、特征是否有效、嫌疑集合是否可被非成员污染、以及在灰盒访问下效果如何。

最核心的结论是：CDI 能在 8 个扩散模型上给出稳定的集合级判断。对 `U-ViT256-T2I-Deep` 这类 COCO 文本条件模型，只需约 `70` 个嫌疑样本即可达到 `p<0.01`。对训练集更大的 ImageNet 模型，则需要更多样本。作者进一步总结出三条趋势：训练集越大越难识别，输入分辨率越高越容易识别，训练步数越多信号越强。

消融同样很关键。去掉统计检验后，set-level MIA 的 `TPR@FPR=1%` 在多个模型上只有 `6.50%`、`10.20%`、`23.20%` 这一量级，而完整 CDI 可提升到 `74.43%`、`24.92%`、`100.00%`。特征消融说明新增特征显著降低了达到显著性所需的最小样本数，例如 `U-ViT512` 从约 `20000` 个样本降到约 `2000` 个样本。论文还显示，当 `P` 与 `U` 都由非成员组成时，平均 `p` 值约为 `0.38` 到 `0.40`，说明方法不会系统性地产生假阳性。

在更现实的灰盒访问下，作者删除了依赖梯度或内部优化的白盒特征，仅保留已有 MIA 特征与 `Multiple Loss`。即便如此，CDI 仍然可以拒绝零假设，只是平均需要比白盒多约三分之一的样本。这一结果对实际部署尤为关键，因为它说明 CDI 在失去最强特征后仍保留可用性，但样本预算会明显上升。

## 6. 局限与可复现性

论文的首要限制是它强依赖同分布的未公开集合 `U`。在真实版权争议中，创作者未必拥有足够大的未公开样本集，或者这些样本与公开作品的分布差异较大，这会直接削弱统计检验的可解释性。

第二个限制是访问假设。商业闭源文生图系统通常不提供任意扩散步的噪声预测，更不允许读取梯度，因此 CDI 最容易复现的仍是开放权重模型。论文虽然给出了统一开源实现，但若要忠实复现实验，仍需要目标模型接口、`P/U` 数据组织、5 折训练、1000 次重采样以及较大规模推理预算。

## 7. 对 DiffAudit 的直接启发

这篇论文对 DiffAudit 最直接的价值，是把“单样本灰盒成员推断”扩展成“集合级版权审计”。如果仓库后续要支持创作者级或图库级审计交付，CDI 提供了更合适的输出语言：不是某张图的分数，而是整组作品是否显著高于同分布对照。

在实现路径上，论文建议的最短路线并不是先追求白盒特征，而是先把现有灰盒 `SecMI`、`PIA` 一类结果标准化成可拼接特征，再叠加 `Multiple Loss`、评分模型和显著性检验。这样可以在不改变当前灰盒基础设施的前提下，尽快补上“集合级证据聚合”这一层。
