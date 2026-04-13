# 结构记忆驱动的文生图扩散模型成员推断精修笔记
Unveiling Structural Memorization: Structural Membership Inference Attack for Text-to-Image Diffusion Models

## 文档说明

- GitHub PDF：[2024-arxiv-structural-memorization-membership-inference-text-to-image-diffusion.pdf](https://github.com/DeliciousBuding/DiffAudit-Research/blob/main/references/materials/gray-box/2024-arxiv-structural-memorization-membership-inference-text-to-image-diffusion.pdf)
- 对应报告：[论文报告：Unveiling Structural Memorization: Structural Membership Inference Attack for Text-to-Image Diffusion Models](https://www.feishu.cn/docx/CQ1VdhIhxoowmbxW1qWc9a6FnTd)
- 开源实现：暂未找到
- 整理说明：本稿基于同目录 born-digital Markdown 精修，保留原论文的核心公式、实验数字和章节逻辑，但压缩为便于复核的中文笔记

## 摘要精修

论文关注的是文生图扩散模型中的成员推断。作者认为，已有方法大多依赖像素级噪声误差，默认模型会以逐像素方式记住训练图像；但对于在 LAION 级别数据上训练的大模型，这个假设过强。更现实的情况是，模型更可能记住图像的结构，尤其是布局、轮廓和大尺度语义骨架。

基于这一判断，论文先分析图像在前向扩散过程中的结构演化，再提出一种结构式成员推断攻击。具体做法是把输入图像送入目标文生图模型的 encoder，得到 latent 表示后执行 DDIM inversion，将带噪 latent 再解码回图像空间，最后用原图与输出图的 SSIM 作为成员分数。实验表明，该方法在 Latent Diffusion Model 和 Stable Diffusion v1-1 上都优于 `SecMI`、`PIA`、`Naive Loss`，并且对额外噪声和轻度图像变换更稳健。

## 1. 研究问题与核心观察

论文真正想验证的是：文生图扩散模型的记忆是否主要体现在结构层，而非像素层。作者先从扩散过程本身入手，指出前向扩散在初期主要破坏局部细节，而整体结构在较早阶段仍能保留；只有到后期，结构才会明显坍塌。若模型对成员图像存在结构记忆，那么成员在同样的前向扩散步数下，应当比非成员保留更多结构。

作者据此比较 member 与 hold-out 图像在不同时间步上的结构相似度变化，并观察到两点。第一，non-member 的结构相似度在最初约一百个扩散步下降得更快。第二，member 与 hold-out 的平均 SSIM 差值会在大约 `t=100` 附近达到峰值。这两个观察共同决定了攻击器的时间步选择和得分定义。

## 2. 预备知识与关键公式

论文沿用扩散模型的标准前向过程

$$
q(x_{1:T}|x_0)=\prod_{t=1}^{T}q(x_t|x_{t-1}), \qquad
q(x_t|x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t\mathbf{I}),
$$

以及 DDIM inversion 的确定性更新

$$
x_{t+1}=\sqrt{\alpha_{t+1}}\left(\frac{x_t-\sqrt{1-\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\alpha_t}}\right)+\sqrt{1-\alpha_{t+1}}\,\epsilon_\theta(x_t,t).
$$

在文生图场景中，图像先被编码到 latent 空间，前向扩散与 inversion 也在 latent 中进行，然后再解码回图像空间。这样做的好处是避免直接在像素空间比较噪声纹理，而是让模型自己的 latent 轨迹决定结构如何退化。

## 3. 结构演化分析

论文用 SSIM 衡量原图和带噪图之间的结构相似度，并定义下降速度

$$
v(t)=\frac{\operatorname{SSIM}(x_0,x_{t+\Delta t})-\operatorname{SSIM}(x_0,x_t)}{\Delta t}.
$$

这个量用于刻画“结构在某一段扩散区间内被破坏得有多快”。实验结果显示，在最早的一段扩散过程中，hold-out 图像的 `v(t)` 更负，也就是结构损失更快。作者进一步定义 member 与 hold-out 的平均结构差

$$
\Delta \operatorname{SSIM}(t)=\frac{1}{|X_m|}\sum_{x_0\in X_m}\operatorname{SSIM}(x_0,x_t)-\frac{1}{|X_h|}\sum_{x_0\in X_h}\operatorname{SSIM}(x_0,x_t),
$$

并观察到这条曲线在约 `t=100` 达到峰值。也就是说，若想让 member / non-member 最可分，应当在早期扩散阶段取样，而不是把图像推到太深的噪声区间。

## 4. 结构式成员推断攻击

在上述观察基础上，攻击器本身非常简单。给定查询图像，先用目标模型 encoder 得到 latent 表示；再用 BLIP 生成文本提示，并结合 DDIM inversion 把 latent 推到目标时间步；之后将带噪 latent 解码回图像空间；最后计算原图与输出图的 SSIM，并和阈值比较：

$$
\hat{m}(x_0)=\mathbb{1}\!\left[\operatorname{SSIM}(x_0,x_t)>\tau\right].
$$

这里的关键不是又训练了一个额外分类器，而是找到更合适的统计量。作者强调，像素级噪声误差更容易受到额外噪声和图像轻度扰动的影响，而结构相似度直接作用在图像空间，因而对真实世界失真更稳健。文本条件方面，论文并不假设已知训练 caption，而是默认用 BLIP 生成近似文本，这让方法更接近实际审计场景。

## 5. 实验设置与主要结果

实验使用两个目标模型：Latent Diffusion Model 和 Stable Diffusion v1-1；每个模型都抽取 `5000` 张成员图像，并用 `5000` 张 COCO2017-Val 图像作为 hold-out。评估分辨率为 `512x512` 和 `256x256`，基线包括 `SecMI`、`PIA`、`Naive Loss`，指标包括 `AUC`、`ASR`、`Precision`、`Recall` 与低误报区间 TPR。

最能说明问题的是 `512x512` 主结果：

| 模型 | 方法 | AUC | ASR | TPR@1%FPR | TPR@0.1%FPR |
| --- | --- | --- | --- | --- | --- |
| LDM | Ours | 0.930 | 0.860 | 0.575 | 0.245 |
| LDM | Naive Loss | 0.789 | 0.740 | 0.338 | 0.231 |
| SD v1-1 | Ours | 0.920 | 0.852 | 0.512 | 0.234 |
| SD v1-1 | Naive Loss | 0.766 | 0.717 | 0.310 | 0.215 |

论文还给出两个重要补充结论。第一，总扩散步数 `T=100` 最优，继续增大到 `400`、`600`、`800` 会明显损伤可分性，说明信号集中在早期结构衰减区间。第二，在 LDM `512x512` 的扰动实验里，附加噪声下本文方法仍有 `AUC=0.710`，显著高于 `SecMI` 的 `0.566` 和 `PIA` 的 `0.399`，这和“结构级分数更抗噪”这一方法直觉相一致。

## 6. 复现与使用注意事项

若要复现实验，需要目标文生图模型的 encoder / decoder / U-Net 推理链路、可运行的 DDIM inversion、BLIP caption 生成器、member 子集和 COCO hold-out 集，以及和论文一致的时间步、采样间隔和阈值校准方式。论文正文没有提供官方代码仓库，因此很多实现细节需要从正文与补充材料逆推。

从 DiffAudit 的角度看，这篇论文更适合作为灰盒对照实现，而不是直接进入严格黑盒主线。它最值得复用的不是某个特定超参数，而是“把成员信号定义为前向扩散中的结构保持性”这一思路。只要仓库后续具备稳定的 latent 前向扩散接口，就可以把这条结构分数链路与 `SecMI`、`PIA` 放到统一框架下做对比。
