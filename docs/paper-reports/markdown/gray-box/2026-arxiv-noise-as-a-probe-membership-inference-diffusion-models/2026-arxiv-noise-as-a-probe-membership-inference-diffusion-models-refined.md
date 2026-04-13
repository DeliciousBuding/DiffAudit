# 初始噪声作为探针：利用起始噪声对扩散模型做成员推断（精修版）
Noise as a Probe: Membership Inference Attacks on Diffusion Models Leveraging Initial Noise

## 文档说明

- 作者：Puwei Lian，Yujun Cai，Songze Li，Bingkun Bao
- 发表信息：arXiv 预印本，2026，`arXiv:2601.21628v1`
- 威胁模型：gray-box，text-to-image diffusion membership inference with controllable initial noise
- GitHub PDF：[2026-arxiv-noise-as-a-probe-membership-inference-diffusion-models.pdf](https://github.com/DeliciousBuding/DiffAudit-Research/blob/main/references/materials/gray-box/2026-arxiv-noise-as-a-probe-membership-inference-diffusion-models.pdf)
- 本地 PDF：`D:\Code\DiffAudit\Research\references\materials\gray-box\2026-arxiv-noise-as-a-probe-membership-inference-diffusion-models.pdf`
- born-digital 源稿：[2026-arxiv-noise-as-a-probe-membership-inference-diffusion-models.md](./2026-arxiv-noise-as-a-probe-membership-inference-diffusion-models.md)
- 对应展示稿：[论文报告：Noise as a Probe: Membership Inference Attacks on Diffusion Models Leveraging Initial Noise](https://www.feishu.cn/docx/KjgqdFESSoivAlxKfjyc1JxJn8X)
- 开源实现：论文称将公开实现，但当前 PDF 未给出仓库链接
- 整理说明：本稿基于 born-digital 原文精修，优先保留论文主线、关键公式、主结果与对 DiffAudit 有直接影响的边界条件，便于继续细修或后续索引引用。

## 1. 论文定位

这篇论文是典型的 gray-box 成员推断工作，但它故意避开了现有扩散模型 MIA 最常见的一条路径：读取中间去噪结果。作者关注的不是“如何更好地从 denoising trajectory 里读信号”，而是“如果系统只暴露最终生成接口，攻击者还能利用什么”。他们给出的答案是初始噪声。

因此，这篇论文在路线上的位置不是 `SecMI`、`PIA` 的轻量变体，而是 gray-box 主线的一次接口重定义。它把审计焦点从中间层观测转移到采样起点，适合作为 gray-box 向更受限黑盒环境延伸时的桥接论文。

## 2. 核心问题

作者真正想验证的是两个观察能否同时成立。第一，扩散模型在最大噪声步并没有把原图语义彻底清空，初始噪声中仍保留可利用的残留信息。第二，微调模型会在训练过程中学会利用这些残留信息，因此当初始噪声与训练成员相关时，最终生成结果会更接近原图，从而暴露成员身份。

这个问题与以往工作不同的地方在于，攻击信号不再来自中间网络输出或额外训练的 shadow classifier，而来自“模型面对带语义起始噪声时的最终响应差异”。

## 3. 威胁模型与前提

论文设定下，攻击者拥有候选数据集 \(D\) 及其文本条件 \(c\)，但不知道 member / non-member 划分。攻击者可以调用目标微调模型做端到端生成，可以显式修改初始噪声和 prompt，只能看到最终生成图像，不能访问模型参数，也不能读取中间去噪步。

作者还要求攻击者能访问目标模型对应的预训练底座，或至少能拿到足够接近的预训练模型来执行 DDIM inversion。这一点很关键，因为论文的 semantic injection 并不是直接由目标模型完成，而是先依赖预训练模型提取语义化初始噪声，再交给目标模型做最终生成。

## 4. 方法总览

方法可以概括为“预训练 inversion + 目标模型重建 + 距离判别”。先对候选样本 \(x_0\) 与 prompt \(c\) 使用预训练模型做 DDIM inversion，得到带有原图语义的初始噪声 \(\tilde{x}_t\)。再把 \(\tilde{x}_t\) 和同一文本条件送入目标模型，生成结果 \(\tilde{x}_0\)。最后，比较 \(\tilde{x}_0\) 与原图 \(x_0\) 的距离，小于阈值则判为成员。

作者声称这条路线成立的根本原因有三点：噪声日程在 \(T\) 步仍保留非零信号；模型在微调时学会利用这种残留语义；微调后的模型与其预训练底座在语义空间上仍高度相近，因此预训练 inversion 得到的语义噪声仍能驱动目标模型重建。

## 5. 方法概览 / 流程

论文 Figure 3 把这条方法链路表达得很清楚：左侧是 inversion，把原图压回语义化初始噪声；右侧是 generation，从这份噪声重新采样并计算最终距离。这个流程的重要性在于它不要求攻击者能把噪声注入到中间时间步，而只要求控制起始噪声张量。

对 DiffAudit 来说，这张图最重要的价值不是视觉总结，而是帮助后续实现做模块拆分：`inversion`、`generation with custom noise`、`distance scoring` 三个步骤可以各自做成独立组件，而不必绑定任何中间去噪观测逻辑。

## 6. 关键技术细节

作者先从前向扩散公式出发：

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\,\epsilon.
$$

在这个定义下，若最大噪声步真的等于“纯噪声”，那么 \(\bar{\alpha}_T\) 应接近零。作者进一步引入

$$
\mathrm{SNR}(t)=\frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}
$$

来描述残留语义强度，并在 Table 2 中指出常用 schedule 的 \(\mathrm{SNR}(T)\) 均非零，其中 Stable Diffusion 的残留最明显。这为“初始噪声不是严格无语义”提供了定量起点。

接着，作者把 DDIM inversion 写成逐步语义注入过程：

$$
\tilde{x}_{t}=Inv_{\theta}^{t}(x_0 \mid c,\gamma_2),
$$

再把目标模型生成写成

$$
\tilde{x}_0=G_{\theta}(\tilde{x}_t \mid c,\gamma_1).
$$

其中 \(\gamma_2\) 控制 inversion 时的 classifier-free guidance，\(\gamma_1\) 控制生成阶段的 guidance。默认实现使用 \(\gamma_2=1.0\)、inversion 步数 `100`，\(\gamma_1=3.5\)、生成步数 `50`。这些默认值在 Table 6 的超参数分析里被证明并不算敏感。

最终成员推断规则是

$$
\mathcal{A}(x_i,\theta)=\mathbf{1}\!\left[D\!\left(x_0,G_{\theta}(\tilde{x}_t \mid c,\gamma_1)\right)\le\tau\right].
$$

这里默认距离 \(D\) 取 \(\ell_2\)。论文正文清楚表达了“距离越小越像成员”，但附录 Algorithm 1 却把 `Score > τ` 写成成员，这与主文方向矛盾。这个实现歧义必须保留，因为它会直接影响后续复现的阈值方向。

## 7. 实验设置

主实验覆盖四个数据集：Pokemon、T-to-I、MS-COCO、Flickr，对应 member / non-member 规模分别为 `416/417`、`500/500`、`2500/2500`、`1000/1000`。所有模型均基于 `Stable Diffusion v1-4` 微调，分辨率固定 `512`。训练脚本使用 diffusers 官方 fine-tuning 脚本，硬件为单张 `RTX 4090 24GB`。

基线分成两类。上半部分是中间结果攻击：`SecMI`、`PIA` 等；下半部分是端到端攻击：`NA-P`、`GD`、`Feature-T`、`Feature-C`、`Feature-D`。指标采用 `AUC` 与 `TPR@1%FPR`。此外，附录还评估了阈值选择、不同 scheduler、不同 metric、未知预训练架构、caption 缺失、不同推理步数以及防御条件下的表现。

## 8. 主要结果

论文主结果集中在 Table 5。平均来看，该方法达到 `AUC=84.59`、`TPR@1%FPR=18.35`，在所有 end-to-end 攻击里稳定最优。单数据集上，`MS-COCO` 达到 `90.46 / 21.80`，`T-to-I` 达到 `89.24 / 21.60`，说明方法对较强文本条件数据集尤其有效。

这张主结果表能直接支持两条判断。第一，在不访问中间结果、也不训练 shadow model 的前提下，方法明显优于既有端到端基线。第二，它在某些场景已经接近甚至超过 `SecMI`、`PIA` 这类中间结果攻击，说明“起始噪声接口”本身就足以承载较强成员信号。

附加实验同样重要。Table 7 表明，与 naive 随机噪声方案相比，语义化初始噪声让平均 AUC 再提高 `21.57%`，平均 `TPR@1%FPR` 提高 `10.63%`。Table 8 显示在 `SSei` 与数据增强防御下，所有方法都会退化，但该方法依然保持组内最强。Table 9 说明即便不知道目标模型的精确预训练架构，只要使用较近似的 SD 系列或 SDXL 系列底座做 inversion，攻击仍然有效。

## 9. 优点

这篇工作的第一个优点是重新定义了信号入口。它没有继续沿着“更精细地利用中间去噪结果”这条老路线推进，而是识别出一个在部署接口里更常见、也更容易被忽视的泄露面。第二，论文的三条 observation 之间连接较紧，不是只靠主表数字堆结论。第三，附录补得比较完整，尤其是阈值、scheduler、caption 缺失和 defense 分析，让读者能看见方法在不同约束下如何退化。

## 10. 局限与有效性威胁

最显著的限制是访问假设。很多商用服务并不允许用户显式传入初始噪声，因此论文的攻击前提未必能直接转移到线上闭源 API。其次，方法依赖预训练底座和原始或自动生成的 caption；当两者都不可得时，论文并未给出更弱设定下的替代路线。再次，主实验局限于 `SD-v1-4` 微调体系，对更现代闭源系统的外推仍需谨慎。

还有两个实现层面的有效性威胁也需要保留。其一，正文与附录对阈值方向的符号不一致。其二，作者在 PDF 中未给出公开代码链接，因此一些操作细节只能从文字和图表反推，复现成本并不低。

## 11. 对 DiffAudit 的价值

对 DiffAudit 来说，这篇论文最值得保留的不是某个单一结果数值，而是它清楚地定义出另一类 gray-box 接口：如果系统允许操控 latent、seed 或更底层的噪声张量，那么即便没有中间去噪访问，也可以做有效审计。这使它非常适合与 `SecMI`、`PIA` 并列，形成“轨迹访问型灰盒”与“起始噪声访问型灰盒”的对照结构。

工程上，它提示后续实现不必一开始就依赖内部网络输出，而可以先搭起更短的三段式流程：预训练 inversion、目标模型 custom-noise generation、最终距离评分。产品叙事上，它也帮助解释一件很关键的事：隐私风险不只存在于模型参数和中间层，同样可能存在于采样入口。

## 12. 关键图使用方式

本稿当前保留两张图，但分工不同。Figure 3 服务方法理解，帮助把流程拆成可实现模块；Table 5 服务结果判断，展示它在 end-to-end 组内的主导性。如果后续需要进一步压缩展示稿，优先保留 Table 5；如果需要增强工程可读性，则优先保留 Figure 3。

## 13. 复现评估

忠实复现至少需要以下资产：`SD-v1-4` 微调模型、对应预训练底座、带 caption 的 member / non-member 划分、自定义初始噪声接口、DDIM inversion 实现、非成员分位数阈值校准集，以及可选的 cross-attention 可视化工具。附录中的阈值选择策略是利用一批先验非成员样本取第 `15` 百分位作为判定阈值，因此还需要额外准备校准集，而不只是测试集。

DiffAudit 当前如果只具备常规 gray-box MIA 的推理骨架，还不足以直接复现这篇论文，因为这里额外需要把 inversion、noise injection 和最终结果度量串成统一管线。真正的结构性阻塞不是论文读不懂，而是很多真实目标系统根本不暴露初始噪声接口，这会让方法只能在本地开放权重环境中验证。

## 14. 写回总索引用摘要

这篇论文研究微调扩散模型中的成员推断问题，但不再依赖中间去噪结果，而是把攻击入口转移到初始噪声。作者指出，常见噪声日程在最大噪声步仍保留非零语义，因此“初始噪声”可以被重新解释为成员推断探针。

论文提出的核心方法是先用公开预训练模型对候选样本做 DDIM inversion，得到带语义的起始噪声，再将该噪声送入目标微调模型，根据生成结果与原图的距离做成员判定。实验结果表明，这一路线在多个数据集上都显著优于既有 end-to-end 攻击，并在部分场景接近传统灰盒基线。

对 DiffAudit 而言，这篇工作最重要的价值是明确提出了“可控起始噪声”这一类 gray-box 接口。它既是现有轨迹访问型灰盒工作的补充，也为后续受限部署环境下的审计实现提供了更短路径的工程框架。
