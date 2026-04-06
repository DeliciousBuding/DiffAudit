# 小噪声注入驱动的扩散模型成员推断精修笔记

Noise Aggregation Analysis Driven by Small-Noise Injection: Efficient Membership Inference for Diffusion Models

## 文档说明

- GitHub PDF：[2025-arxiv-small-noise-injection-membership-inference-diffusion-models.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/gray-box/2025-arxiv-small-noise-injection-membership-inference-diffusion-models.pdf)
- 对应展示稿：[Noise Aggregation Analysis Driven by Small-Noise Injection: Efficient Membership Inference for Diffusion Models](https://www.feishu.cn/docx/QswEdNkWKoHj5YxeAXKcusROnHh)
- 开源实现：论文正文未给出公开代码仓库
- 整理说明：本稿基于同目录 born-digital Markdown 与本地阅读报告精修，优先保留威胁模型、方法逻辑、关键数字和对 DiffAudit 的落点

## 摘要精修

这篇论文研究的是 gray-box 扩散模型成员推断，不是输出级黑盒攻击。作者的核心判断是：如果只在待测图像上注入很小的噪声，成员样本在若干相邻时间步上的噪声预测会更稳定、更聚集，非成员样本则更分散。于是，成员性可以被转写成“局部时间邻域内的噪声聚合度”。

方法上，作者不再沿完整前向扩散链逐步走到目标时间步，而是利用前向扩散闭式形式一次性构造目标时刻的带噪样本。随后，攻击者在少量相邻时间步上读取去噪网络的噪声预测，并用 L1、L2、质心距离、平均密度或凸包体积度量这些预测是否足够集中。论文声称该策略在 DDPM 上以 5 次查询取得优于 SecMI 的 ASR 和 AUC，但在 Stable Diffusion 上的低误报率指标并不占优。

## 1. 问题定义与威胁模型

作者要回答的问题不是“扩散模型会不会泄露成员信息”，而是“能否用更少查询稳定提取成员信号”。既有方法里，NaiveLoss 之类的直接信号往往不够强，SecMI 虽然有效，但需要更长的查询链。本文希望在查询次数、信号强度和实现复杂度之间找到更短路径。

威胁模型属于 gray-box。攻击者不需要读取模型参数或梯度，但必须能够在给定时间步上查询目标扩散模型的噪声预测，并知道扩散调度、时间步定义以及 DDIM 式的确定性回推过程。这比纯黑盒生成 API 强得多，因此不应把该文结论直接外推到闭源图像生成服务。

## 2. 方法主线

方法分成三步。

第一步是小噪声注入。对原始图像 `x_0`，作者不走多步前向扩散，而是直接利用闭式形式构造目标时间步 `t` 对应的带噪样本 `x_t`。这一步的意图是保留足够多的原始结构，同时让模型进入一个可以读出成员差异的局部邻域。

第二步是局部时间邻域去噪。攻击者把 `x_t` 输入去噪网络，得到时刻 `t` 的噪声预测，再按 DDIM 式的确定性回推构造 `t-m, t-2m, ...` 等相邻时间步上的噪声预测序列。论文强调，真正有用的不是某个单点预测，而是这组相邻预测的几何聚合程度。

第三步是把噪声序列转换成成员分数。作者把所有相邻时间步上的噪声向量收集成一个集合，再计算其聚合指标；越集中，越可能是成员；越发散，越可能是非成员。默认最好用的是 L2 average distance。

## 3. 关键公式与直觉

前向扩散的关键闭式形式是

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I).
$$

这一步把“逐步把样本推到时间步 `t`”改写成“一次性注入目标方差的噪声”。这样做的直接收益是减少查询次数，也让攻击更聚焦于局部邻域而不是整条长链。

成员分数则写成

$$
S_m = -\log\!\left(C(E) + \delta\right),
$$

其中 `E` 是相邻时间步的噪声预测集合，`C(E)` 是聚合度量。直觉上，如果成员样本的噪声预测在局部时间邻域内更稳定，那么 `C(E)` 更小，`S_m` 更高，更容易被判成成员。

论文还用一个更偏解释性的判断说明其方法成立基础：成员样本对应的条件噪声预测熵更低，因此在轻微扰动下更容易保持局部一致性。这不是严格证明攻击一定成功，而是给出“为什么噪声聚合可能携带成员性”的解释框架。

## 4. 实验结果摘录

主实验落在 DDPM 的 CIFAR-10、CIFAR-100 和 Tiny-ImageNet 上，额外扩展到 Stable Diffusion v1.4/v1.5。最值得保留的数字有三类。

| 场景 | 关键结果 |
| --- | --- |
| CIFAR-10 DDPM | 作者报告 `ASR=0.901`、`AUC=0.957`、`TPR@1%FPR=28.7`，均高于 SecMI |
| CIFAR-100 DDPM | `ASR=0.839`、`AUC=0.903`，但低误报率指标增幅已明显收缩 |
| Stable Diffusion | `ASR/AUC` 仍优于对比方法，但 `TPR@1%FPR` 仅约 `8`，反而落后于 NaiveLoss |

这些结果说明，方法在标准 DDPM 上的确有竞争力，尤其在 CIFAR-10 上最强；但在 latent diffusion 大模型上，论文自己的数据已经显示它不是全面优势路线。

作者还做了参数消融。最关键的结论是：噪声不是越大越好。噪声标准差过小，不足以放大成员和非成员之间的差异；过大则破坏原图语义结构，使局部聚合关系失真。论文给出的经验最佳点大约在 `sigma = 0.1` 附近。另一个经验结论是去噪步数 `k=5` 最稳，继续增大反而会让局部聚合关系被拉散。

## 5. 方法边界

这篇论文最需要保留的限制有三点。

第一，威胁模型偏强。攻击者必须能在指定时间步上拿到噪声预测，这在大多数真实 API 场景下并不成立，所以它更适合作为研究型 gray-box 方法，而不是现实黑盒基线。

第二，大模型证据不够稳。论文虽然强调可扩展到 Stable Diffusion，但它自己的低误报率指标已经暴露出方法在 latent diffusion 上并不稳定，因此不能把 DDPM 结论直接外推到文本到图像大模型。

第三，复现信息不完整。正文没有给出公开代码仓库，Stable Diffusion 部分的接口细节、阈值校准和成员/非成员划分也写得不够充分，因此忠实复现仍会面临实验漂移。

## 6. 对 DiffAudit 的落点

这篇论文最适合被归到 gray-box 路线中的 aggregation-based 分支。它与 loss-based、posterior-based 路线不同，强调的是局部时间邻域上的噪声一致性，而不是单步 loss 或重建误差。

对 DiffAudit 而言，它至少有三层价值。第一，它给出一个非常明确的“查询效率也应成为攻击设计目标”的研究方向。第二，它把成员性信号落到了一个可替换、可比较的聚合器接口上，这对仓库后续抽象攻击配置很有帮助。第三，它也提醒我们不要把 DDPM 上的好结果过度推广到 latent diffusion 或更弱访问权限场景。

更稳妥的落点表述应该是：这是一条查询成本较低、接口要求较强、在 DDPM 上证据较好但在 latent diffusion 上低误报率不足的 gray-box 路线。
