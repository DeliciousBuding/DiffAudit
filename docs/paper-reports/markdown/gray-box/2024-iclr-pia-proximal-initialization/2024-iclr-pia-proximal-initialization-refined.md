# 近邻初始化驱动的扩散模型成员推断精修笔记
An Efficient Membership Inference Attack for the Diffusion Model by Proximal Initialization

## 文档说明

- GitHub PDF：[2024-iclr-pia-proximal-initialization.pdf](https://github.com/DeliciousBuding/DiffAudit-Research/blob/main/references/materials/gray-box/2024-iclr-pia-proximal-initialization.pdf)
- 本地 PDF：`<DIFFAUDIT_ROOT>/Research/references/materials/gray-box/2024-iclr-pia-proximal-initialization.pdf`
- born-digital 源稿：[2024-iclr-pia-proximal-initialization.md](./2024-iclr-pia-proximal-initialization.md)
- 对应展示稿：[论文报告：An Efficient Membership Inference Attack for the Diffusion Model by Proximal Initialization](https://www.feishu.cn/docx/VPNhdJMryo5K86xOfE8cX3PFnDd)
- 开源实现：[kong13661/PIA](https://github.com/kong13661/PIA)
- 整理说明：本稿基于本地 born-digital markdown 精修，保留公式、主结果和 DiffAudit 落地相关信息，优先服务本地阅读与后续细修

## 1. 论文定位

`PIA` 是灰盒扩散模型成员推断路线中的关键推进论文。它沿着 `SecMI` 已经证明有效的“中间时间步可见”设定继续往前走，但把多步迭代求轨迹的思路改成两次查询的近邻初始化攻击，因此更适合作为工程上真正可执行的基线。

## 2. 核心问题

论文要回答的是：当攻击者只能访问扩散模型的中间噪声或 score 输出时，是否可以仅凭 `t=0` 与目标时刻 `t` 两次查询，就判断样本是否属于训练集，同时把攻击推广到连续时间模型和音频扩散模型。

## 3. 威胁模型与前提

攻击者知道待测样本 `x_0`，可查询 `t=0` 和时刻 `t` 的模型输出，但看不到模型参数和梯度。这个设定比白盒弱，但仍明显强于只返回最终图像的黑盒 API。阈值选择通常需要辅助划分或 surrogate model，因此论文证明的是灰盒接口上的成员可分性，而不是完全无先验的在线攻击。

## 4. 方法总览

论文的关键观察是：在 DDIM 的确定性轨迹里，只要知道 `x_0` 和轨迹上任意一点 `x_k`，就能恢复任意 `x_t`。作者于是直接取最靠近原样本的 `k=0`，用模型在 `t=0` 的输出 `\epsilon_\theta(x_0,0)` 近似真实噪声，再构造 `x_t` 并发起第二次查询。成员样本若更贴近训练时学到的轨迹，那么两次查询结果之间的一致性应更强，故攻击分数更小。

这张方法图把 `PIA` 的真正创新点表达得很清楚：第一次查询的价值在于提供 proximal initialization，而不是像 Naive Attack 那样只产生一次性的 loss 信号。

## 5. 方法概览 / 流程

实际执行链可以压缩成四步：先查询 `t=0` 得到 `\epsilon_\theta(x_0,0)`；再将其与原样本组合得到时刻 `t` 的点 `x_t`；随后再次查询目标模型得到在 `t` 的噪声预测；最后计算两次输出的 `\ell_p` 距离并与阈值比较。这种设计让 `PIA` 的主线复杂度接近一个双查询统计器，而不是需要反复迭代的攻击流程。

## 6. 关键技术细节

轨迹恢复的核心公式是

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\cdot\frac{x_k-\sqrt{\bar{\alpha}_k}x_0}{\sqrt{1-\bar{\alpha}_k}}.
$$

选择 `k=0` 后，`x_k` 等价于由 `\epsilon_\theta(x_0,0)` 定义的近邻点，于是离散时间攻击统计量化简为

$$
R_{t,p}=
\left\|
\epsilon_{\theta}(x_0,0)-
\epsilon_{\theta}\!\left(
\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon_{\theta}(x_0,0),\, t
\right)
\right\|_p.
$$

这个式子本质上比较的是“第一次查询得到的近邻初始化”和“第二次查询在该轨迹点上的预测输出”之间的差距。成员样本若被模型拟合得更好，那么这两者更一致，`R_{t,p}` 就更小。

论文还提出了归一化版本 `PIAN`：

$$
\hat{\epsilon}_{\theta}(x_0,0)=N\sqrt{\frac{\pi}{2}}\frac{\epsilon_{\theta}(x_0,0)}{\|\epsilon_{\theta}(x_0,0)\|_1}.
$$

它试图把 `t=0` 的噪声预测拉回近似标准高斯尺度，但这一步只是启发式修正。后文实验表明，这个修正对 DDPM 和 Grad-TTS 有时有帮助，但在 Stable Diffusion 上可能直接失效。

## 7. 实验设置

图像离散时间模型部分使用 CIFAR10、CIFAR100、TinyImageNet 上的 DDPM；latent diffusion 部分使用 Stable Diffusion v1.5，并以 Laion-Aesthetics v2.5+ 为成员集、COCO2017-val 为留出集；连续时间与音频部分使用 Grad-TTS，并额外对 DiffWave、FastDiff 做鲁棒性比较。基线包含 Naive Attack、`SecMI`、`PIA`、`PIAN`，指标以 AUC 和 `TPR@1%FPR` 为主。

## 8. 主要结果

论文最醒目的结果出现在 Grad-TTS：`PIA` 在 LJSpeech 上达到 `99.6` 的 AUC、`94.2` 的 `TPR@1%FPR`；在 LibriTTS 上，`PIAN` 的 `TPR@1%FPR` 进一步提升到 `44.7`。DDPM 上，`PIA` 相对 `SecMI` 的优势主要体现在低误报区间，例如 CIFAR100 上从 `11.1` 提升到 `19.6` 的 `TPR@1%FPR`。Stable Diffusion 上则出现明显分化：`PIA` 仍优于 `SecMI`，但 `PIAN` 大幅退化，说明归一化技巧不适合 latent diffusion。

这张图虽然来自附录，但它直接展示了低 FPR 区间下 `PIA` 对 Naive Attack 与 `SecMI` 的优势，因此比单纯复述表格均值更适合用于本地展示稿和路线判断。

## 9. 优点

- 两次查询即可完成攻击，显著降低灰盒成员推断的时间成本。
- 统一覆盖离散时间与连续时间扩散模型。
- 报告了更贴近审计需求的低 FPR 指标，而非只看平均 AUC。
- 把音频扩散模型纳入成员推断讨论，补足了既有工作主要聚焦视觉的空缺。

## 10. 局限与有效性威胁

- 攻击依赖中间输出接口，现实闭源服务不一定暴露这种能力。
- 连续时间推导存在近似项，因此更偏工程统计量而非严格等价推导。
- `PIAN` 缺乏稳固理论支持，并在 Stable Diffusion 上被实验性否定。
- 音频模型“更鲁棒”的结论当前仍主要是经验观察。

## 11. 对 DiffAudit 的价值

对 DiffAudit 来说，`PIA` 是当前灰盒路线里最适合落地的默认主线之一。它保留了 `SecMI` 所依赖的中间输出接口，但把查询成本大幅压低，因此非常适合作为仓库中灰盒 runtime probe、smoke test 和后续真实资产接入的主力基线。

## 12. 关键图使用方式

本稿暂保留两张图。方法图用于解释 proximal initialization 如何替代 `SecMI` 的迭代回推；结果图用于强调 `PIA` 的收益主要发生在低误报区间，而不是仅靠平均 AUC 撑结论。

## 13. 复现评估

从当前仓库状态看，`src/diffaudit/attacks/pia.py` 已实现 `pia` 计划与资产探测逻辑，`tests/test_pia_adapter.py` 也覆盖了 runtime probe 与 synthetic smoke，这说明 DDPM 侧的最短路径已经存在。真正缺的仍是论文全量实验矩阵：Stable Diffusion 的成员划分、Grad-TTS / DiffWave / FastDiff 的完整资产，以及连续时间接口与阈值扫描流程。

## 14. 写回总索引用摘要

这篇论文研究扩散模型上的灰盒成员推断，重点是在只能访问 `t=0` 和目标时刻 `t` 中间输出的条件下，用尽可能少的查询判断样本是否属于训练集。

作者提出 `PIA` 与 `PIAN`。核心做法是把模型在 `t=0` 的输出当作 proximal initialization，用它构造 groundtruth trajectory，再与时刻 `t` 的预测结果计算 `\ell_p` 距离。实验表明，该方法在 DDPM、Stable Diffusion 与 Grad-TTS 上通常能达到与 `SecMI` 相当或更高的 AUC，并在低误报区间表现更好。

它对 DiffAudit 的价值在于提供了一条更轻量、更易工程化的灰盒主线，而且与当前仓库已有的 `pia` 计划器、探针和 smoke 逻辑直接对齐。
