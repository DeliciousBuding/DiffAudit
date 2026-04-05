# 无需 Caption 的成员推断：基于模型拟合嵌入的无文本审计（精修版）
No Caption, No Problem: Caption-Free Membership Inference via Model-Fitted Embeddings

## 文档说明

- 作者：Joonsung Jeon，Woo Jae Kim，Suhyeon Ha，Sooel Son，Sung-Eui Yoon
- 发表信息：ICLR 2026 conference paper
- 威胁模型：gray-box，caption-free text-to-image diffusion membership inference
- GitHub PDF：[2026-openreview-mofit-caption-free-membership-inference.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/gray-box/2026-openreview-mofit-caption-free-membership-inference.pdf)
- 本地 PDF：`D:\Code\DiffAudit\Project\references\materials\gray-box\2026-openreview-mofit-caption-free-membership-inference.pdf`
- born-digital 源稿：[2026-openreview-mofit-caption-free-membership-inference.md](./2026-openreview-mofit-caption-free-membership-inference.md)
- 对应展示稿：[2026-openreview-mofit-caption-free-membership-inference-report.md](../../../gray-box/2026-openreview-mofit-caption-free-membership-inference-report.md)
- 开源实现：[JoonsungJeon/MoFit](https://github.com/JoonsungJeon/MoFit)
- 整理说明：本稿基于 born-digital markdown 精修，保留论文主线、关键公式、主结果与附录里最影响判断的边界条件。

## 1. 论文定位

MOFIT 是一篇 gray-box 成员推断论文，目标对象是 text-to-image latent diffusion model。它相对 CLiD、PIA、PFAMI 这类工作的真正增量不是“攻击更强”本身，而是把成员推断从“默认拿到 ground-truth caption”的理想化设定推进到“只有图像、没有真实文本标注”的现实设定。按 DiffAudit 路线划分，它应归入 gray-box 主线中的 caption-free 子分支。

## 2. 核心问题

论文要解决的问题可以写成一句话：若审计者只有查询图像 \(x\)，却拿不到训练时配对 caption \(c\)，还能否恢复足够强的条件信号来区分 member 与 hold-out。作者先用实验说明，直接把真实 caption 换成 VLM caption 会显著破坏 CLiD 的分离性，然后进一步追问：有没有一种不依赖真实文本、但能比 VLM caption 更贴近目标模型内部分布的替代条件。

## 3. 威胁模型与前提

攻击者可访问目标 LDM 的去噪网络与中间损失，在指定扩散步上计算 \(L_{\text{cond}}\) 与 \(L_{\text{uncond}}\)，并能对图像扰动和条件嵌入做梯度优化。攻击者看不到真实 caption，但允许调用 VLM 生成 caption 作为初始化。该设定仍然属于灰盒，因为核心能力是读取并优化模型内部损失；它不适用于只能通过最终图像输出做重建误差比较的纯黑盒环境。

## 4. 方法总览

作者的关键观察是“成员样本对错配条件更敏感”。当 ground-truth caption 被 VLM caption 替换时，member 的 \(L_{\text{cond}}\) 明显上升，而 hold-out 的变化较小；与此同时，\(L_{\text{uncond}}\) 对 member 往往更低。MOFIT 就围绕这个观察设计两阶段流程：先把图像推到模型内部更熟悉的无条件流形，再从这个 surrogate 提取一个模型拟合条件嵌入，最后故意把该嵌入配回原图以制造失配。

## 5. 方法概览 / 流程

第一阶段是 model-fitted surrogate optimization。给定查询图像 \(x_0\)，作者优化扰动 \(\delta\)，使 \(x_0+\delta\) 在 null condition 下的无条件去噪损失尽可能低。第二阶段是 surrogate-driven embedding extraction。作者把条件嵌入 \(\phi\) 当成可优化变量，在同一个 \(t\) 与 \(\hat{\epsilon}\) 下最小化 surrogate 的条件损失，得到 \(\phi^\*\)。最终用原图 \(x_0\) 与 \(\phi^\*\) 计算条件损失，再与无条件损失做差并联合辅助分数判定成员身份。

## 6. 关键技术细节

surrogate 优化对应论文公式 (5)：

$$
\delta^\* := \arg\min_{\delta}\; \mathbb{E}_{z'_0,t,\hat{\epsilon}} \left[\left\|\hat{\epsilon}-\epsilon_\theta(z'_t,t,\phi_{\text{null}})\right\|^2\right].
$$

这一步的作用不是生成更“自然”的图，而是让 surrogate 更贴近目标模型学到的 unconditional prior。作者强调在优化时固定时间步和目标噪声，以稳定扰动方向，并让 surrogate 沿模型内部更低损失的方向收敛。

embedding 提取对应公式 (6)：

$$
\phi^\* := \arg\min_{\phi}\; \mathbb{E}_{z_0^\*,t,\hat{\epsilon}} \left[\left\|\hat{\epsilon}-\epsilon_\theta(z_t^\*,t,\phi)\right\|^2\right].
$$

这一步并不试图恢复真实 caption 的语义文本，而是构造一个与 surrogate 在模型内部高度耦合的条件变量。作者用 VLM caption embedding 作为初始化，但最终得到的是 model-fitted embedding，而不是普通 caption embedding。

最终主分数是

$$
\mathcal{L}_{\text{MoFit}}=
\mathbb{E}_{z_0,t,\hat{\epsilon}}\left[\left\|\hat{\epsilon}-\epsilon_\theta(z_t,t,\phi^\*)\right\|^2\right]
-
\mathbb{E}_{z_0,t,\hat{\epsilon}}\left[\left\|\hat{\epsilon}-\epsilon_\theta(z_t,t,\phi_{\text{null}})\right\|^2\right].
$$

推断时作者再把 \(\mathcal{L}_{\text{MoFit}}\) 与 \(-L_{\text{aux}}\) 做 robust scaling 后线性组合。这里的关键机制是：\((x_0^\*,\phi^\*)\) 是模型内高度过拟合的一对，而 \((x_0,\phi^\*)\) 是故意构造出的错配对；正是这个错配把 member 对条件变化的高敏感性放大成可分离信号。

## 7. 实验设置

主实验使用 Stable Diffusion v1.4 在 Pokemon、MS-COCO、Flickr 上的微调模型。基线包括 Loss、SecMI、PIA、PFAMI 和 CLiD；在 caption-free 条件下，这些基线统一改用 VLM 生成 caption。真实场景数据集 MS-COCO 与 Flickr 用 BLIP-2 生成 caption，Pokemon 用 CLIP-Interrogator。作者固定 \(t=140\)、\(T=1000\)，surrogate 阶段用梯度符号更新 \(\delta\)，embedding 阶段用 Adam，指标是 ASR、AUC、TPR@1%FPR。

## 8. 主要结果

主结果表如下，直接体现了“VLM caption 不够，model-fitted embedding 才够”的论点：

| 方法 | 条件 | Pokemon ASR / AUC / TPR@1%FPR | MS-COCO ASR / AUC / TPR@1%FPR | Flickr ASR / AUC / TPR@1%FPR |
|---|---|---|---|---|
| CLiD | GT | 96.52 / 99.17 / 90.14 | 86.50 / 90.27 / 68.80 | 91.10 / 95.13 / 77.20 |
| CLiD | VLM | 77.55 / 83.43 / 19.23 | 80.90 / 86.53 / 50.80 | 79.00 / 85.16 / 40.60 |
| MOFIT | \(\phi^\*\) | 94.48 / 97.30 / 50.48 | 88.00 / 94.17 / 47.00 | 86.00 / 91.32 / 53.20 |

对 Pokemon，MOFIT 几乎追回了 GT-captioned CLiD 的 ASR/AUC，但高精度区 TPR 仍明显落后。对 MS-COCO，MOFIT 的 ASR/AUC 甚至超过 GT-captioned CLiD，但 TPR@1%FPR 不是最优，这说明它更像是在整体排序上恢复了分离性，而不是在所有 operating point 上都最强。对 Flickr，MOFIT 同样全面压过所有 VLM-captioned 基线。

## 9. 优点

这篇工作的优点主要有三点。第一，威胁模型更现实，真正面向“只有图像没有文本”的审计方。第二，方法不是经验拼装，而是由一个清晰观察驱动：member 对条件失配更敏感。第三，论文不只报主结果，还专门检查 surrogate 的作用、LoRA 的影响、运行时成本和 early stopping，因此读者能更准确判断方法边界。

## 10. 局限与有效性威胁

局限也很明确。其一，方法需要可微访问损失并做两阶段优化，访问假设偏强。其二，所谓 caption-free 实际上是不依赖真实 caption，而不是完全去掉文本侧先验，因为初始化还用到了 VLM embedding。其三，附录显示 LoRA 场景下 MOFIT 与大多数基线都接近随机水平。其四，默认运行成本高，作者给出的总时间大约是每张图 7 到 9 分钟。其五，预训练 SD v1.5 的评估并未沿用标准 LAION-mi member split，而是改成 431 个 verified memorized samples，因此相关结果更适合说明“在强成员信号存在时方法可迁移”，而不是直接代表标准预训练审计难度。

## 11. 对 DiffAudit 的价值

对 DiffAudit 来说，MOFIT 最重要的价值是补齐 gray-box 路线里最现实的一块空白：真实 caption 缺失时如何继续审计。它非常适合作为 CLiD 的对照论文一起进入主线，一前一后说明两个层次的问题：CLiD 证明条件过拟合可用来做成员推断，MOFIT 证明即便拿不到真实文本，也可以通过 surrogate 与拟合嵌入重建足够强的信号。

## 12. 关键图使用方式

下面这张图对应论文 Figure 3，直接展示了 MOFIT 的可分离性来源：member 在 \(\phi^\*\) 条件下的 \(L_{\text{cond}}\) 分布明显上移，而 hold-out 变化有限，因此 \(\mathcal{L}_{\text{MoFit}}\) 的正向偏移主要发生在 member 一侧。

![](./_page_7_Figure_11.jpeg)

这张图适合放在讲结果机制的位置，而不是单纯作为“漂亮配图”。它能帮助我们判断以后在仓库中复现实验时该优先记录哪些统计量：至少要保留 \(L_{\text{cond}}\)、\(L_{\text{uncond}}\) 以及最终 score 的 member/hold-out 分布。

## 13. 复现评估

从复现角度看，这篇论文并不缺方法细节，但缺的是低成本实现路径。需要的资产包括目标模型权重、member/hold-out 划分、VLM 初始化器、可读内部损失并支持反向优化的推理接口，以及阈值与 \(\gamma\) 的标定数据。仓库目前如果只具备普通评测脚本，还不能直接复现 MOFIT，因为 surrogate 优化和 embedding 提取都需要单样本迭代式优化。结构性阻塞则主要来自吞吐量和访问权限，而不是缺论文信息。

## 14. 写回总索引用摘要

这篇论文研究的是文生图 latent diffusion model 在没有真实 caption 时的成员推断。它指出，既有方法一旦把真实 caption 换成 VLM caption，成员与非成员的条件损失分布会重新重叠，攻击效果明显下降。

为了解决这个问题，作者提出 MOFIT：先在无条件分支上优化查询图像得到 model-fitted surrogate，再从 surrogate 中提取与其过拟合耦合的条件嵌入，并故意将其错配回原图，以放大成员样本对条件失配的敏感性。主实验显示，MOFIT 在三个微调 SD v1.4 模型上都显著优于 VLM-captioned 基线，并在 MS-COCO 上超过了 GT-captioned CLiD 的 ASR/AUC。

对 DiffAudit 而言，这篇工作应被视为 caption-free gray-box 主论文。它不仅扩展了攻击设定，也给出了具体的两阶段实现范式，同时明确暴露了访问假设强、运行成本高、LoRA 下退化明显等边界。
