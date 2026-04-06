# 扩散模型是否易受成员推断攻击？
Are Diffusion Models Vulnerable to Membership Inference Attacks?

## 文献信息

- 英文标题：Are Diffusion Models Vulnerable to Membership Inference Attacks?
- 中文标题：扩散模型是否易受成员推断攻击？
- 作者：Jinhao Duan，Fei Kong，Shiqi Wang，Xiaoshuang Shi，Kaidi Xu
- 发表 venue / year / version：ICML 2023，PMLR 202
- 论文主问题：当攻击者能够访问扩散过程某个时间步的中间去噪状态或其等价误差信号时，是否可以可靠地区分训练成员与非成员样本
- 威胁模型类别：灰盒成员推断攻击
- 本地 PDF 路径：`D:/Code/DiffAudit/Project/references/materials/gray-box/2023-icml-secmi-membership-inference-diffusion-models.pdf`
- GitHub PDF 链接：[2023-icml-secmi-membership-inference-diffusion-models.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/gray-box/2023-icml-secmi-membership-inference-diffusion-models.pdf)
- OCR/Markdown 精修版链接：[OCR精修版：Are Diffusion Models Vulnerable to Membership Inference Attacks?](https://www.feishu.cn/docx/BsckdH7uNoHemwx1HfocgJhsnbg)
- 飞书原生 PDF：[2023-icml-secmi-membership-inference-diffusion-models.pdf](https://ncn24qi9j5mt.feishu.cn/file/Oc72bykWdoE45MxU2FKcBTYQnQe)
- 开源实现：[jinhaoduan/SecMI](https://github.com/jinhaoduan/SecMI)
- 报告状态：已完成

## 1. 论文定位

这篇论文是扩散模型成员推断方向的早期代表性工作，核心贡献不是简单把已有生成模型攻击迁移到扩散模型上，而是先证明既有 GAN/VAE 路线大多失效，再据此提出适配扩散训练目标的灰盒攻击 `SecMI`。因此，它在 DiffAudit 的灰盒路线中更适合作为起点论文和基准论文，用来定义“扩散模型成员信号应当观察什么”。

## 2. 核心问题

论文集中回答两个问题。第一，扩散模型是否真的像表面现象那样“不容易做成员推断”，还是只是旧方法没有抓住扩散模型的泄露接口。第二，如果成员泄露确实存在，最有效的信号究竟来自最终生成样本、重构误差，还是来自某个时间步上的后验估计误差。作者的结论是：扩散模型并不天然安全，关键在于利用 step-wise posterior estimation error，而不是沿用传统生成模型的距离型或判别器型攻击。

## 3. 威胁模型与前提

论文采用灰盒设定。攻击者持有待测样本 `x_0`，能够在选定时间步 `t_{\mathrm{sec}}` 上调用目标扩散模型的 deterministic reverse 与 deterministic denoise 过程，或者至少拿到与之等价的中间状态，从而计算单样本 `t-error`。攻击者不需要白盒梯度权限，也不要求知道真实训练集划分；但 `SecMI_NNs` 需要少量 member / hold-out 样本训练攻击分类器。该结论只适用于存在中间时间步可观测接口的系统；若实际产品只返回最终图像，这篇论文的方法不能直接套用。

## 4. 方法总览

`SecMI` 的直觉非常直接。扩散模型在训练时逐步拟合前向扩散后验，如果一个样本属于训练集，那么模型在某些时间步上应当更准确地逼近对应后验，表现为更小的估计误差。作者据此把成员推断问题改写为误差比较问题：先在时间步 `t_{\mathrm{sec}}` 上执行一次 deterministic reverse，再做一次 deterministic denoise，比较往返后的误差是否足够小。若误差显著偏小，则更可能是训练成员。

论文给出两个攻击版本。`SecMI_stat` 直接对 `t-error` 做阈值判断，目的是验证信号本身是否存在；`SecMI_NNs` 则把像素级绝对误差图送入一个轻量分类器，以提升判别精度。与已有方法相比，这篇论文真正改变的是攻击观测量：不再依赖最终生成样本与成员样本的相似度，而是直接对准扩散模型训练目标里的局部后验拟合误差。

## 5. 方法概览 / 流程

方法流程可以概括为三步。首先，给定待测样本并选取固定时间步 `t_{\mathrm{sec}}`；其次，通过 deterministic reverse 获得该时间步的中间状态，再通过 deterministic denoise 回推一个相邻状态；最后，把往返前后的偏差计算为 `t-error`，并用阈值规则或神经网络规则输出成员判定。论文还指出，较小的时间步通常保留更多原样本信息，因此成员与非成员的误差差异往往更明显。

## 6. 关键技术细节

方法的理论起点是扩散模型在单步上的 posterior matching 目标：

$$
\ell_t = \mathbb{E}_q \left[\frac{1}{2\sigma_t^2}\left\| \tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t) \right\|^2 \right].
$$

这条式子说明，模型在每个时间步都在最小化真实后验均值与参数化估计之间的偏差。若成员样本在训练中被拟合得更充分，那么同一时间步上的局部估计误差应当系统性更小，这就是论文把成员推断落到单步误差比较上的原因。

由于真实 posterior estimation error 难以直接解析计算，作者引入 deterministic reverse 与 deterministic denoise 做近似，定义单样本、单时间步的 `t-error`：

$$
\tilde{\ell}_{t,x_0} = \left\| \psi_\theta\!\left(\phi_\theta(\tilde{x}_t, t), t\right) - \tilde{x}_t \right\|^2.
$$

这个量可以理解为“把样本推进到第 `t` 步后，再走一轮可逆近似，模型还能否稳定回到原先状态”。如果模型对该样本更熟悉，这个往返误差就更小。论文在实验里进一步验证了一个经验事实：当 `t` 接近 0 时，成员与非成员的 `t-error` 差距通常更大，因为此时状态中保留了更多与原样本相关的细节。

最简单的判定规则是阈值攻击：

$$
\mathcal{M}(x_0,\theta) = \mathbf{1}\!\left[\tilde{\ell}_{t_{\mathrm{sec}},x_0} \le \tau \right].
$$

这对应 `SecMI_stat`。而 `SecMI_NNs` 则把像素级绝对误差图输入攻击分类器，从“单个标量是否足够”推进到“误差空间分布能否提供更多成员线索”。这一区分很重要，因为它把“信号存在性”与“信号利用效率”拆开了：前者由阈值法证明，后者由神经网络法放大。

## 7. 实验设置

标准扩散模型部分，作者在 DDPM 上评估 CIFAR-10、CIFAR-100、STL10-U 和 Tiny-ImageNet。文本到图像部分，作者进一步评估 LDM 与 Stable Diffusion，其中 LDM 使用 Pokemon 与 COCO2017-val，Stable Diffusion 使用 Laion-aesthetic-5plus 与 COCO2017-val 的成员 / 非成员划分。指标采用 `ASR`、`AUC`、`TPR@1%FPR` 与 `TPR@0.1%FPR`，并固定 `t_{\mathrm{sec}} = 100`，以 `DDIM(k)` 加速推断。`SecMI_NNs` 的攻击模型采用 ResNet-18，并使用少量 member / hold-out 样本监督训练。

基线方面，论文先系统重评了 LOGAN、TVD、Over-Representation、Monte-Carlo Set 和 GAN-Leaks 等既有生成模型成员推断方法，再与 `SecMI_stat`、`SecMI_NNs` 对比。这一设计的价值在于，它先把“旧方法为什么不行”说明白，再证明新方法有效，避免把结果解释成单纯的工程调参收益。

## 8. 主要结果

论文最关键的结论有三点。第一，已有 GAN/VAE 路线在 DDPM 上大多接近随机，说明“扩散模型上旧方法失效”并不等于“扩散模型不存在成员泄露”。第二，`SecMI` 在 DDPM 上给出稳定而明显的可分性：四个数据集上的平均 `ASR/AUC`，`SecMI_stat` 为 `0.810/0.881`，`SecMI_NNs` 为 `0.889/0.949`。第三，在严格低误报区间，方法仍有可用信号，例如 `SecMI_NNs` 在四个 DDPM 数据集上的 `TPR@1%FPR` 最高达到 `37.98%`，`TPR@0.1%FPR` 最高达到 `7.59%`。

扩展到文本到图像模型后，攻击仍然有效，但强度明显下降。LDM 在 Pokemon 和 COCO2017-val 上的 `AUC` 分别可达 `0.891` 与 `0.875`；而 Stable Diffusion v1-4 / v1-5 的 `AUC` 分别为 `0.707` 与 `0.701`。这说明 `SecMI` 不仅适用于小规模 DDPM，也能迁移到更现实的生成系统，但攻击效果会随着模型规模、条件复杂度和侧信息质量而弱化。

![SecMI 在 DDPM 上的 ROC 曲线](../assets/gray-box/2023-icml-secmi-membership-inference-diffusion-models-key-figure-p7.png)

这张 ROC 图同时展示了总体可分性与低 FPR 区间表现。对团队展示来说，它比单纯列均值更有说服力，因为它直接说明 `SecMI` 并不是只在宽松阈值下有效，而是在高置信度判断区间仍保留成员信号。

## 9. 优点

这篇论文的首要优点是问题定义准确。作者没有把扩散模型简单并入“通用生成模型 MIA”，而是明确指出攻击应与扩散训练目标对齐。第二，实验设计完整，既有对旧方法失效的系统复评，也有对 DDPM、LDM、Stable Diffusion 的跨模型验证。第三，指标选择较为扎实，低 FPR 区间结果使论文不只停留在“平均上可分”，而是更接近实际风险讨论。

## 10. 局限与有效性威胁

这篇论文最大的限制是访问假设偏强。`SecMI` 依赖中间时间步的 reverse / denoise 结果，而现实服务通常只暴露最终图像。第二，`SecMI_NNs` 需要少量带标签的 member / hold-out 数据训练攻击器，因此严格来说并非纯无监督攻击。第三，论文对防御的论证还不充分；例如 DP-SGD、强正则和 RandAugment 导致 DDPM 训练不收敛，这更接近“模型没训练好”，而不是对有效模型上的防御效果做公平比较。第四，文本到图像实验显示 prompt 质量会显著影响结果，说明侧信息可得性是重要边界条件。

## 11. 对 DiffAudit 的价值

对 DiffAudit 而言，这篇论文的价值主要体现在三层。第一，它是灰盒路线的主论文，用来定义最典型的可观测接口，即中间时间步误差信号。第二，它给出了一个很清晰的路线分层基准：如果系统拿不到中间状态，就不能直接复用 `SecMI`，而需要转向更弱访问假设的方法。第三，它对产品叙事也有帮助，因为它提醒我们不要把强灰盒结果直接外推成“所有扩散 API 都存在同等风险”，而应明确说明风险成立所依赖的访问条件。

## 12. 关键图使用方式

本报告保留 1 张图，选择 ROC 主结果图而不是方法图。原因是团队展示更需要先回答“这篇论文到底证没证明扩散模型存在成员泄露”，而 ROC 图同时覆盖总体性能与低误报区间，更能支撑这一主结论。方法流程在正文中已经可用文字说明清楚，因此没有额外加入流程图。

## 13. 复现评估

若做忠实复现，至少需要目标扩散模型权重、严格的 member / hold-out 划分、deterministic reverse / denoise 推理链路、固定的 `t_{\mathrm{sec}}` 与 `DDIM(k)` 设置，以及文本到图像场景下可用的 prompt 或替代 prompt。真正的结构性阻塞不在分类器本身，而在是否具备与论文一致的中间状态访问接口。若这一接口不存在，就只能做方法改造而不是论文原样复现。对 DiffAudit 来说，更务实的起步顺序是先在 DDPM 小模型上跑通 `t-error` 计算与阈值版攻击，再决定是否扩展到 LDM 或 Stable Diffusion。

## 14. 写回总索引用摘要

这篇论文研究扩散模型上的成员推断问题，重点不是最终生成样本是否接近训练集，而是攻击者能否利用扩散过程某个时间步上的后验估计误差来识别成员样本。

作者提出 `SecMI`，通过 deterministic reverse 与 deterministic denoise 构造 `t-error`，并给出阈值版 `SecMI_stat` 和学习版 `SecMI_NNs`。结果表明，旧生成模型 MIA 在扩散模型上大多失效，但基于 step-wise posterior estimation error 的攻击在 DDPM、LDM 和 Stable Diffusion 上都能得到明显高于随机的成员可分性。

它对 DiffAudit 的价值在于为灰盒路线提供了第一篇可作为基准的主论文：一方面定义了需要什么中间观测接口，另一方面也明确了这一路线的边界，即强访问假设下效果较强，但不能直接等同于现实黑盒产品风险。
