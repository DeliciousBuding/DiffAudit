# DiffAudit在线索引

**GitHub 仓库主页**：[DeliciousBuding/DiffAudit](https://github.com/DeliciousBuding/DiffAudit/)

**更新时间**：
2026-04-06 04:03:17 +08:00

**文档用途**：供团队内部查看当前研究范围、复现状态、关键阻塞，以及每篇材料的详细阅读报告入口。当前版本已切换到研究归档与展示版口径。

**同步基线提交**：
830850c

## 当前状态概览
| 模块 | 当前阶段 | 当前判断 | 主要阻塞 | 下一步 |
| --- | --- | --- | --- | --- |
| 黑盒 | runtime evidence expanding | 黑盒主线已不止停留在文档刷新，`recon` 已打通 `Stable Diffusion + DDIM` 与 `kandinsky_v22` 的最小真实 runtime-mainline，`DiT` 官方 sample smoke 也已通。 | 公开资产仍只覆盖极小子集，target/shadow/member/non-member 语义映射尚未最终核准。 | 扩大 `recon` 公开子集规模，补统一状态文档与在线索引同步。 |
| 灰盒 | quality-refresh in progress | 灰盒条目已全部建档；其中 SecMI、Structural Memorization、CDI 与 MoFit 四篇已按新规范完成报告、精修原文与 PDF 三件套收口，其余灰盒条目仍沿用旧批次文档。 | 旧批次文档风格和互链尚未统一。 | 继续按同一流程刷新其余灰盒论文，并把方法差异映射到统一实验计划。 |
| 白盒 | research-ready | 白盒主论文与解释型路线已完成详细阅读报告，研究脉络已经明确。 | 缺 checkpoint、梯度/激活接口与复现实验资产。 | 先补资产，再决定优先复现哪条白盒线。 |
| 防御/综述 | research-ready + indexed | 综述、防御和扩展方向材料已形成完整阅读归档，可直接支撑团队选题与对外叙事。 | 尚未全部映射为仓库实验路线。 | 按优先级挑选进入下一轮实验实现。 |

## 阅读报告进度
- 目录层面：`27 / 27` 篇论文均已有单篇飞书文档链接。
- 新规范刷新进度：黑盒 `4 / 4` 已完成按新标准重发。
- 当前可直接阅读的黑盒三件套已全部落入各自飞书目录，包含 `报告 + 精修原文 + PDF`。
- 新规范刷新进度：灰盒首批 `4` 篇已完成按新标准重发，分别是 `SecMI`、`Structural Memorization`、`CDI` 与 `MoFit`。
- 说明：这里区分“已有文档”与“按新规范刷新完成”两种状态，避免把旧草稿误记为已达标展示稿。

## 当前主线
当前实验主线仍然是黑盒方向中的微调扩散模型成员推断，但仓库状态已经从单纯文档整理推进到真实运行证据积累阶段。当前最强证据来自 `recon` 的公开最小子集：`Stable Diffusion + DDIM` 与 `kandinsky_v22` 都已经打通 runtime-mainline，`DiT` 则处于官方 sample smoke 已通、尚未进入真实 benchmark 的状态。

## 关键入口
- 复现状态总览：[docs/reproduction-status.md](https://github.com/DeliciousBuding/DiffAudit/blob/main/docs/reproduction-status.md)
- 黑盒统一结果摘要：[experiments/blackbox-status/summary.json](https://github.com/DeliciousBuding/DiffAudit/blob/main/experiments/blackbox-status/summary.json)
- 黑盒主线计划：[workspaces/black-box/plan.md](https://github.com/DeliciousBuding/DiffAudit/blob/main/workspaces/black-box/plan.md)
- 阅读报告清单：[docs/paper-reports/manifest.csv](https://github.com/DeliciousBuding/DiffAudit/blob/main/docs/paper-reports/manifest.csv)
- 论文索引源文件：[references/materials/paper-index.md](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/paper-index.md)

## 论文索引说明
- 每篇条目统一给出文件位置、内容简介、核心方法 / 结论、与 DiffAudit 的关系、开源实现，以及单篇阅读报告入口。
- `内容简介 / 核心方法 / 结论 / 和 DiffAudit 的关系` 三栏均由详细阅读报告中的 Extracted Summary 自动回写。
- 在线文档中的本地 PDF 链接统一使用 GitHub 浏览链接；`阅读报告` 链接统一指向对应飞书文档。

## 论文在线索引

## 黑盒

### Towards Black-Box Membership Inference Attack for Diffusion Models

- 文件：[2024-arxiv-towards-black-box-membership-inference-diffusion-models.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/black-box/2024-arxiv-towards-black-box-membership-inference-diffusion-models.pdf)
- 内容简介：这篇论文研究扩散模型的黑盒成员推断问题，目标是在看不到 U-Net、DiT 或中间 timestep 输出的情况下，仅凭图像变体 API 判断某张图像是否出现在训练集中。论文将该问题定位为版权审计和训练数据使用取证的现实需求，尤其面向商业化闭源扩散服务。
- 核心方法 / 结论：作者提出 `REDIFFUSE`，核心做法是在固定扩散步下多次调用 variation API，对返回图像求平均后再与原图比较；若平均重构更接近原图，则更可能是训练成员。论文同时给出平均化误差收缩的理论解释，并在 DDIM、Stable Diffusion、Diffusion Transformer 上报告了优于 Loss、SecMI、PIA、PIAN 等基线的结果。
- 和 DiffAudit 的关系：它对 DiffAudit 的价值在于明确提供了一条独立的黑盒路线：当目标系统只暴露公开生成接口而不暴露内部噪声估计时，仍然可以围绕 query budget、差异函数和代理分类器设计成员审计流程。相较当前仓库已覆盖的 `SecMI`、`PIA` 灰盒链路，这篇论文更适合作为未来黑盒实验与产品叙事的基础文献。
- 开源仓库：暂未找到
- 阅读报告：[Towards Black-Box Membership Inference Attack for Diffusion Models](https://www.feishu.cn/docx/C8qcdxXjpoK3VyxMCzGcxUaOnXg)

### Membership Inference on Text-to-Image Diffusion Models via Conditional Likelihood Discrepancy

- 文件：[2024-neurips-clid-membership-inference-text-to-image-diffusion.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/black-box/2024-neurips-clid-membership-inference-text-to-image-diffusion.pdf)
- 内容简介：这篇论文讨论文本到图像扩散模型的成员推断，核心问题是判断一条图文样本是否参与过模型训练。作者指出，已有扩散模型成员推断方法主要依赖图像侧误差或高成本 shadow models，在文本到图像模型上会受到更强泛化能力和扩散训练随机性的影响，因此在更真实的训练步数与默认数据增强条件下效果明显下降。
- 核心方法 / 结论：论文提出 Conditional Likelihood Discrepancy（CLiD），其出发点是文本到图像模型对条件分布 `p(x|c)` 的过拟合强于对边缘分布 `p(x)` 的过拟合。作者用完整文本、空文本和多种削弱文本条件之间的 ELBO 近似差值构造成员性指标，并实现了阈值版 `CLiDth` 与向量版 `CLiDvec`。在 Pokemon、MS-COCO、Flickr 微调场景以及一个处理后的 LAION 预训练场景中，CLiD 在 AUC、ASR 和 `TPR@1%FPR` 上普遍优于 Loss、SecMI、PIA 和 PFAMI 等基线，且对默认数据增强更稳健。
- 和 DiffAudit 的关系：对 DiffAudit 而言，这篇论文最重要的价值是证明“条件扰动前后的响应差异”是比纯图像误差更强的审计信号。不过论文本身采用的是灰盒查询设定，需要访问扩散过程中的中间噪声预测，因此不能直接当作严格 black-box 方法复用。它更适合作为 black-box 路线的邻近理论支点和信号设计参考。
- 开源仓库：[zhaisf/CLiD](https://github.com/zhaisf/CLiD)
- 阅读报告：[Membership Inference on Text-to-image Diffusion Models via Conditional Likelihood Discrepancy](https://www.feishu.cn/docx/KkHfdqntFoDyrnxIjhNc2d1InZb)

### Black-box Membership Inference Attacks against Fine-tuned Diffusion Models

- 文件：[2025-ndss-black-box-membership-inference-fine-tuned-diffusion-models.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/black-box/2025-ndss-black-box-membership-inference-fine-tuned-diffusion-models.pdf)
- 内容简介：这篇论文研究黑盒成员推断在微调扩散模型上的可行性，核心问题是在仅有文本到图像查询接口、且攻击者无法访问参数与中间状态的条件下，判断给定样本是否属于目标模型的微调训练集。作者将威胁模型聚焦于下游微调数据，而非预训练数据，并把攻击场景分成“是否拥有原始提示词”和“辅助数据是否与成员集重叠”两个维度下的四种设置。
- 核心方法 / 结论：论文提出的核心方法是基于相似度分数的黑盒攻击框架：对同一查询多次调用目标模型，比较查询图像与生成图像在特征空间中的相似度，再通过阈值、分布检验或分类器输出成员判定。论文报告在 CelebA-Dialog、WIT、MS COCO 上取得较高 AUC，并在有限采样预算下明显优于既有黑盒基线；同时，DeiT 与余弦相似度最稳定，DP-SGD 会显著削弱攻击效果。
- 和 DiffAudit 的关系：对 DiffAudit 而言，这篇论文是黑盒路线中的关键参照，因为它证明了仅凭输出接口与有限辅助信息，微调扩散模型就可能泄露训练参与信息。它还明确揭示了黑盒路线中的真实边界条件，例如 captioning 质量、辅助数据重叠、影子模型选择和采样预算，因此适合用作黑盒审计叙事与实验分层的锚点文献。
- 开源仓库：[py85252876/Reconstruction-based-Attack](https://github.com/py85252876/Reconstruction-based-Attack)
- 阅读报告：[Black-box Membership Inference Attacks against Fine-tuned Diffusion Models](https://www.feishu.cn/docx/UJeGd68rJolI66xW7P1czimDnHb)

### Membership Inference Attacks for Face Images Against Fine-tuned Latent Diffusion Models

- 文件：[2025-visapp-membership-inference-face-fine-tuned-latent-diffusion-models.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/black-box/2025-visapp-membership-inference-face-fine-tuned-latent-diffusion-models.pdf)
- 内容简介：这篇论文研究的是一个面向人脸图像的黑盒成员推断问题：当 Stable Diffusion v1.5 被某批人脸照片微调后，外部攻击者能否仅通过 prompt 查询与生成结果，判断这批人脸是否属于训练集。作者特别把问题限制在 face-image fine-tuning 场景，因此结论更像是高敏感领域案例分析，而不是对所有扩散模型的普遍断言。
- 核心方法 / 结论：论文的方法是先用目标模型生成图像，把这些生成图像当作成员分布的正样本代理，再配合同域 auxiliary negatives 训练一个 ResNet-18 攻击模型。结果表明，该方法在数据集级别通常有效，例如 DTU 与 AAU 人脸集合之间可获得约 `0.86` 的 AUC；visible watermark 会把 AUC 推高到 `1.00`，guidance scale 也会显著影响结果；但对同一分布中的 seen / unseen 单样本识别基本失败，AUC 约为 `0.53`。
- 和 DiffAudit 的关系：它对 DiffAudit 的意义在于补充了“面向敏感人脸数据的黑盒审计”这一现实叙事，并揭示了生成条件、辅助数据构造和水印会显著改变成员信号强度。仓库当前已有黑盒主线骨架，但尚未覆盖这篇论文所需的人脸抓取、BLIP 标注、SD v1.5 微调与攻击分类器训练，因此该文更适合作为黑盒路线的案例参照和条件分析材料。
- 开源仓库：[osquera/MIA_SD](https://github.com/osquera/MIA_SD)
- 阅读报告：[Membership Inference Attacks for Face Images Against Fine-Tuned Latent Diffusion Models](https://www.feishu.cn/docx/A9VLdD2eioc5IcxKq9zcb0HOnxg)

## 灰盒

### Are Diffusion Models Vulnerable to Membership Inference Attacks?

- 文件：[2023-icml-secmi-membership-inference-diffusion-models.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/gray-box/2023-icml-secmi-membership-inference-diffusion-models.pdf)
- 内容简介：这篇论文研究扩散模型上的成员推断问题，目标是在攻击者掌握待测样本并可访问扩散过程部分中间信号时，判断该样本是否属于训练集。作者首先证明，直接移植 GAN 或 VAE 上的既有生成模型 MIA 到 DDPM 基本无效，因此扩散模型需要单独建模其泄露机制。
- 核心方法 / 结论：作者提出 `SecMI`，核心思想是比较扩散模型在某个时间步上的后验估计误差。论文用 deterministic reverse 与 denoise 近似单样本的 posterior estimation error，定义 `t-error` 作为成员信号，并给出阈值版 `SecMI_stat` 与学习版 `SecMI_NNs`。在 DDPM 四个数据集上，两者平均 `ASR/AUC` 分别达到 `0.810/0.881` 与 `0.889/0.949`，在 LDM 和 Stable Diffusion 上也保持了明显高于随机的可分性。
- 和 DiffAudit 的关系：它对 DiffAudit 的意义在于奠定了灰盒主线的基准接口与证据标准：要获得强成员推断性能，攻击者通常需要接触中间 timestep 的误差信号，而不是只看最终生成结果。当前仓库已经围绕 `SecMI` 准备了 vendored 子集、planner、adapter 与 smoke 流程，因此这篇论文既是灰盒路线的理论起点，也是现有工程骨架最直接对应的基础文献。
- 开源仓库：[jinhaoduan/SecMI](https://github.com/jinhaoduan/SecMI)
- 阅读报告：[Are Diffusion Models Vulnerable to Membership Inference Attacks?](https://www.feishu.cn/docx/ALF5d68CpoIXePxkUUBcnJ0cnPb)

### SIDE: Surrogate Conditional Data Extraction from Diffusion Models

- 文件：[2024-arxiv-side-extracting-training-data-unconditional-diffusion-models.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/gray-box/2024-arxiv-side-extracting-training-data-unconditional-diffusion-models.pdf)
- 内容简介：这篇论文讨论扩散模型训练数据提取中的一个关键误区：无条件扩散模型通常被认为比条件模型安全，因为攻击者没有 prompt 或类别标签来精确引导生成过程。作者的目标是检验这种安全感是否成立，并在统一的理论框架下说明不同形式的条件信息为何会放大记忆与泄露风险。
- 核心方法 / 结论：论文提出 SIDE，通过目标模型自生成样本的特征聚类来构造 surrogate condition，再用时间相关分类器或 LoRA 条件微调把这些伪标签接入反向扩散过程。实验显示，SIDE 在 CIFAR-10、CelebA、ImageNet 和 LAION-5B 上均优于既有无条件与条件抽取基线，说明只要条件足够精确，即便原模型是无条件的，也能被定向推向记忆样本所在的高密度区域。
- 和 DiffAudit 的关系：对 DiffAudit 而言，这篇论文的价值主要在于 threat model 与 side-information 视角的扩展。它不是最贴近当前 gray-box 审计实现的低成本基线，因为主实验要求白盒参数访问，但它揭示了“内部聚类结构可被代理成条件信号”这一更强的泄露机制，可作为后续 gray-box 路线、泄露指标设计和 beyond-membership 训练样本提取叙事的重要背景文献。
- 开源仓库：暂未找到
- 阅读报告：[SIDE: Surrogate Conditional Data Extraction from Diffusion Models](https://www.feishu.cn/docx/TPA1dUmgzoVauDxbPjpcLSz7nOe)

### Unveiling Structural Memorization: Structural Membership Inference Attack for Text-to-Image Diffusion Models

- 文件：[2024-arxiv-structural-memorization-membership-inference-text-to-image-diffusion.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/gray-box/2024-arxiv-structural-memorization-membership-inference-text-to-image-diffusion.pdf)
- 内容简介：这篇论文研究文生图扩散模型上的成员推断攻击，核心问题是：当目标模型由大规模互联网图像训练而成时，攻击是否还能依赖像素级误差识别训练成员。作者认为更现实的信号来自结构级记忆，即训练成员在前向扩散早期比非成员保留更多图像结构，因此成员与非成员在结构相似度衰减曲线上会出现系统性差异。
- 核心方法 / 结论：论文提出一种结构式灰盒攻击。做法是先把输入图像编码到 latent 空间，用 BLIP 生成近似文本提示，再执行 DDIM inversion 得到带噪 latent，最后解码回图像空间，并用原图与输出图的 SSIM 作为成员分数。实验在 Latent Diffusion Model 和 Stable Diffusion v1-1 上表明，该方法在 AUC、ASR、`TPR@1%FPR` 等指标上普遍优于 `SecMI`、`PIA` 和 `Naive Loss`，同时对附加噪声等扰动更稳健。
- 和 DiffAudit 的关系：对 DiffAudit 来说，这篇论文的重要性在于它补足了灰盒路线中的“结构记忆”分支。当前仓库已覆盖 `SecMI` 与 `PIA` 等像素级或噪声级基线，而本文提供了一个不同的比较轴：在相近的灰盒访问条件下，直接利用前向扩散中的结构保持性做成员审计。这使它非常适合作为后续灰盒实验和路线叙事中的对照论文。
- 开源仓库：暂未找到
- 阅读报告：[Unveiling Structural Memorization: Structural Membership Inference Attack for Text-to-Image Diffusion Models](https://www.feishu.cn/docx/CQ1VdhIhxoowmbxW1qWc9a6FnTd)

### An Efficient Membership Inference Attack for the Diffusion Model by Proximal Initialization

- 文件：[2024-iclr-pia-proximal-initialization.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/gray-box/2024-iclr-pia-proximal-initialization.pdf)
- 内容简介：这篇论文研究扩散模型上的灰盒成员推断，目标是在攻击者看不到模型参数、但能访问 `t=0` 与目标时刻 `t` 的中间噪声相关输出时，判断给定样本是否出现在训练集中。作者特别关注查询代价问题，希望用比 `SecMI` 更少的查询获得稳定成员信号。
- 核心方法 / 结论：作者提出 `PIA`（以及归一化版本 `PIAN`），核心做法是把模型在 `t=0` 的输出当作 proximal initialization，用它构造 groundtruth trajectory，再与时刻 `t` 的预测结果做 `\ell_p` 距离比较。论文在 DDPM、Stable Diffusion 与 Grad-TTS 上报告，`PIA/PIAN` 往往能达到与 `SecMI` 相当或更高的 AUC，并在 `TPR@1%FPR` 上更优，同时只需两次查询。
- 和 DiffAudit 的关系：它对 DiffAudit 的意义在于明确给出了灰盒路线里一个更低成本、可工程化的主力方法。当前仓库已经存在 `PIA` 的 planner、资产探测和 smoke 路径，因此这篇论文不仅是灰盒基线文献，也能直接支撑后续真实资产接入与运行链扩展。
- 开源仓库：[kong13661/PIA](https://github.com/kong13661/PIA)
- 阅读报告：[An Efficient Membership Inference Attack for the Diffusion Model by Proximal Initialization](https://www.feishu.cn/docx/Vc9edKDxuo4jghxhweDcXSL6njT)

### Score-based Membership Inference on Diffusion Models

- 文件：[2025-arxiv-sima-score-based-membership-inference-diffusion-models.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/gray-box/2025-arxiv-sima-score-based-membership-inference-diffusion-models.pdf)
- 内容简介：这篇论文研究扩散模型的成员推断攻击，核心问题是攻击者能否仅通过模型对查询样本输出的预测噪声来判断该样本是否属于训练集。作者将目标限定在 gray-box 场景，即攻击者可以访问模型并在给定时间步上读取 denoiser 输出，但不依赖复杂辅助分类器或大量重复查询。
- 核心方法 / 结论：论文提出 SimA，将攻击统计量定义为预测噪声范数 $A(x,t)=\lVert \hat{\epsilon}_\theta(x,t)\rVert_p$，并用高斯卷积后的数据分布、score 函数与有限训练集局部核均值之间的关系解释其有效性。实验表明，SimA 在 DDPM 和 Guided Diffusion 上通常达到或超过既有基线，同时只需单次查询；但在 LDM 与部分 Stable Diffusion 设定上，该类方法整体退化到接近随机，说明潜变量瓶颈显著改变了泄露机制。
- 和 DiffAudit 的关系：对 DiffAudit 而言，这篇论文一方面提供了一个可解释、低查询成本、适合做 gray-box 基线的成员推断方法，另一方面也明确提示 pixel-space 与 latent-space 扩散模型不应被放在同一泄露假设下分析。它既能支撑当前 gray-box 主线的最短路径实现，也能为后续将审计路线拆分到 LDM/Stable Diffusion 提供机制证据。
- 开源仓库：[mx-ethan-rao/SimA](https://github.com/mx-ethan-rao/SimA)
- 阅读报告：[SCORE-BASED MEMBERSHIP INFERENCE ON DIFFUSION MODELS](https://www.feishu.cn/docx/AoQCdwScKof18ExgDxMcDU3znAd)

### Noise Aggregation Analysis Driven by Small-Noise Injection: Efficient Membership Inference for Diffusion Models

- 文件：[2025-arxiv-small-noise-injection-membership-inference-diffusion-models.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/gray-box/2025-arxiv-small-noise-injection-membership-inference-diffusion-models.pdf)
- 内容简介：该文研究扩散模型成员推断攻击，核心问题是在不直接读取模型参数的前提下，判断某张图像是否参与过训练。论文针对现有方法查询成本高、对扩散模型适配性有限的问题，提出利用轻微噪声扰动后噪声预测稳定性差异来识别成员与非成员。
- 核心方法 / 结论：作者的方法先用前向扩散闭式形式向原图一次性注入小噪声，再在相邻时间步上提取去噪网络的噪声预测序列，并用 L1/L2 距离、质心距离、密度或凸包体积衡量其聚合程度。论文报告该方法在 DDPM 的 CIFAR-10、CIFAR-100 和 Tiny-ImageNet 上以 5 次查询取得优于 SecMI 的 ASR 和 AUC；但在 Stable Diffusion 上虽然 ASR/AUC 仍有优势，低误报率指标并不占优。
- 和 DiffAudit 的关系：对 DiffAudit 而言，这篇论文的重要性在于它提供了一条清晰的 gray-box aggregation-based 路线，可与 loss-based、posterior-based 扩散 MIA 方法形成直接对照。它尤其适合被记录为“查询效率较高、但需要时间步噪声预测接口且在 latent diffusion 上证据偏弱”的代表工作。
- 开源仓库：暂未找到
- 阅读报告：[Noise Aggregation Analysis Driven by Small-Noise Injection: Efficient Membership Inference for Diffusion Models](https://www.feishu.cn/docx/QswEdNkWKoHj5YxeAXKcusROnHh)

### CDI: Copyrighted Data Identification in Diffusion Models

- 文件：[2025-cvpr-cdi-copyrighted-data-identification-diffusion-models.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/gray-box/2025-cvpr-cdi-copyrighted-data-identification-diffusion-models.pdf)
- 内容简介：这篇论文研究扩散模型训练数据的版权审计问题，目标不是判定单张图像是否为训练成员，而是判断某个数据拥有者的一组公开作品是否整体被用于训练可疑扩散模型。作者指出，现有针对扩散模型的单样本成员推断在大规模、现实模型上不够稳定，难以单独支撑高置信度版权主张。
- 核心方法 / 结论：论文提出 CDI，将公开嫌疑集合 `P` 与同分布未公开集合 `U` 做对照，从现有 MIA 与三种新增特征中抽取成员性信号，经逻辑回归评分器聚合后，再用单尾 Welch t 检验输出集合级显著性结论。实验显示，CDI 在多类扩散模型上均有效，部分 COCO 文本条件模型只需约 70 个样本即可达到 `p < 0.01`，且统计检验与新特征都会显著降低样本需求。
- 和 DiffAudit 的关系：对 DiffAudit 而言，这篇论文的意义在于提供了一条“数据集级审计 / 证据聚合”路线，适合补强单样本灰盒 MIA 难以解释的场景。它尤其提示：在灰盒设定下，应优先聚合已有 MIA 和 Multiple Loss 这类可访问特征，再叠加统计检验形成更可用于审计叙事的集合级证据。
- 开源仓库：[sprintml/copyrighted_data_identification](https://github.com/sprintml/copyrighted_data_identification)
- 阅读报告：[CDI: Copyrighted Data Identification in Diffusion Models](https://www.feishu.cn/docx/QRzhdNv6NoryLIxPbz7crXbRnd5)

### Noise as a Probe: Membership Inference Attacks on Diffusion Models Leveraging Initial Noise

- 文件：[2026-arxiv-noise-as-a-probe-membership-inference-diffusion-models.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/gray-box/2026-arxiv-noise-as-a-probe-membership-inference-diffusion-models.pdf)
- 内容简介：这篇论文研究微调扩散模型中的成员推断问题，关注点不是中间去噪轨迹，而是初始噪声本身是否残留了足够的语义信息。作者指出，常见噪声日程在最大噪声步仍保留非零信号，因此“初始噪声”并非完全无语义，这为成员推断提供了新的攻击入口。
- 核心方法 / 结论：论文提出的核心方法是先用公开预训练底座对目标样本执行 DDIM inversion，得到带有样本语义的初始噪声，再将该噪声送入目标微调模型并比较生成结果与原图的距离。实验结果表明，这种语义化初始噪声能够明显放大成员与非成员之间的差异，使方法在多个数据集上取得优于既有端到端攻击的 AUC 和低 FPR 指标。
- 和 DiffAudit 的关系：对 DiffAudit 来说，这篇论文的重要性在于它把“可控初始噪声”明确识别为一种可审计接口，说明即便系统不暴露中间结果，仍可能从采样入口和最终生成图像中提取成员信号。它因此是灰盒路线向受限黑盒路线延伸时的关键桥接材料，也提醒后续评估必须关注 seed、latent 或 noise engineering 接口的隐私风险。
- 开源仓库：暂未找到
- 阅读报告：[Noise as a Probe: Membership Inference Attacks on Diffusion Models Leveraging Initial Noise](https://www.feishu.cn/docx/Rga7dRsjdoBu8txA7JrcyvFPnof)

### No Caption, No Problem: Caption-Free Membership Inference via Model-Fitted Embeddings

- 文件：[2026-openreview-mofit-caption-free-membership-inference.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/gray-box/2026-openreview-mofit-caption-free-membership-inference.pdf)
- 内容简介：这篇论文研究 text-to-image latent diffusion models 的成员推断问题，但把攻击条件收紧到更现实的 caption-free 设定：审计者只有查询图像，没有训练时的真实文本标注。作者指出，现有依赖 ground-truth caption 的方法在替换为 VLM 生成 caption 后，成员与非成员的条件损失分布会显著重叠，导致推断性能明显下降。
- 核心方法 / 结论：论文提出 MOFIT。其做法不是恢复真实 caption，而是先在无条件分支上优化扰动，把查询图像推向目标模型学到的先验流形，得到 model-fitted surrogate；再从该 surrogate 中优化出与之紧耦合的条件嵌入 `\phi^\*`。推断时用原图与 `\phi^\*` 的故意失配来放大成员样本的条件损失响应，并用 `L_{\text{MOFIT}}=L_{cond}-L_{uncond}` 及辅助分数做判定。论文报告该方法在 Pokemon、MS-COCO、Flickr 上都显著优于 VLM-captioned 基线，并在 MS-COCO 上超过了使用真实 caption 的 CLiD。
- 和 DiffAudit 的关系：对 DiffAudit 来说，这篇工作的重要性在于它把 gray-box 路线推进到“不依赖真实 caption”的实际审计场景，并提供了一个可复用的两阶段优化框架。它也明确暴露了方法边界：需要可微访问目标模型、计算成本较高、且在 LoRA 场景下效果明显退化，因此既适合作为 caption-free gray-box 主线文献，也适合作为后续防御评估的参照点。
- 开源仓库：[JoonsungJeon/MoFit](https://github.com/JoonsungJeon/MoFit)
- 阅读报告：[No Caption, No Problem: Caption-Free Membership Inference via Model-Fitted Embeddings](https://www.feishu.cn/docx/SuUudTKOSoakt8x9owuc1OhAnqf)

## 白盒

### Finding NeMo: Localizing Neurons Responsible for Memorization in Diffusion Models

- 文件：[2024-neurips-finding-nemo-localizing-memorization-neurons-diffusion-models.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/white-box/2024-neurips-finding-nemo-localizing-memorization-neurons-diffusion-models.pdf)
- 内容简介：这篇论文研究的是扩散模型训练样本记忆的内部定位问题。作者不满足于检测某个 prompt 是否会复现训练图像，而是进一步追问这种记忆是否能够在模型内部被精确归因到少量神经元，目标场景是对公开发布的 text-to-image diffusion model 做强白盒分析与干预。
- 核心方法 / 结论：论文提出 `NEMO` 两阶段算法：先利用非记忆提示词上的激活统计，对记忆提示词中的 cross-attention value neurons 做分布外激活筛选，并加入每层高激活 top-k 神经元形成候选集；再利用首步去噪噪声差异的 `SSIM` 一致性做层级和单神经元精炼。实验报告 verbatim 记忆通常只涉及极少量神经元，中位数为 `4±3`，停用这些神经元后可显著降低 `SSCDGen`，同时基本保持 `ACLIP` 和图像质量。
- 和 DiffAudit 的关系：对 DiffAudit 而言，这篇工作的意义在于把 white-box 路线从“检测模型是否记忆”推进到“定位并干预哪些内部单元在承载记忆”。它为后续的 activation hook、cross-attention 层级消融、定向剪枝和因果验证提供了一个明确框架，也说明扩散模型的训练样本复制并不一定是全局分布式现象，而可能集中在可操作的局部神经元子集上。
- 开源仓库：[ml-research/localizing_memorization_in_diffusion_models](https://github.com/ml-research/localizing_memorization_in_diffusion_models)
- 阅读报告：[Finding NeMo: Localizing Neurons Responsible For Memorization in Diffusion Models](https://www.feishu.cn/docx/G8zqd5tlCoOvGoxTJ4scStxAn0e)

### White-box Membership Inference Attacks against Diffusion Models

- 文件：[2025-local-mirror-white-box-membership-inference-diffusion-models.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/white-box/2025-local-mirror-white-box-membership-inference-diffusion-models.pdf)
- 内容简介：这篇论文研究扩散模型的白盒成员推断问题，目标是在攻击者掌握目标模型参数、结构以及条件模态信息时，判断一个查询样本是否属于训练集。作者把这一问题放在公开 checkpoint 已普遍可得的背景下讨论，并将其视为扩散模型隐私风险分析的强权限场景。
- 核心方法 / 结论：论文提出 `GSA` 框架，用梯度而不是 loss 作为攻击特征，并通过 timestep subsampling 与 layer-wise aggregation 降低维度和成本。作者给出 `GSA1` 与 `GSA2` 两个实例化方法，并报告它们在 DDPM 和 Imagen 上都能取得很高的成员推断精度；在 CIFAR-10 上，`GSA1/GSA2` 的 AUC 达到 `0.999`，明显高于同条件下的 loss-based 对照。
- 和 DiffAudit 的关系：它对 DiffAudit 的价值在于为 white-box 路线提供了强上界参考，同时也明确指出复现该路线所需的关键资产是 checkpoint、训练配置和样本级梯度接口。换言之，这篇论文既是 white-box 方向的重要文献支点，也把后续工程工作的真实阻塞项暴露得很清楚。
- 开源仓库：[py85252876/GSA](https://github.com/py85252876/GSA)
- 阅读报告：[White-box Membership Inference Attacks against Diffusion Models](https://www.feishu.cn/docx/ESTidarzpo2qhux5CkOc7WJVnqd)

### White-box Membership Inference Attacks against Diffusion Models

- 文件：[2025-popets-white-box-membership-inference-diffusion-models.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/white-box/2025-popets-white-box-membership-inference-diffusion-models.pdf)
- 内容简介：这篇论文研究扩散模型的白盒成员推断问题，目标是在攻击者掌握目标模型参数、结构以及条件模态信息时，判断某个查询样本是否属于训练集。作者把这一问题放在公开 checkpoint 已普遍可得的现实背景下讨论，并将其视为扩散模型隐私风险分析的强权限场景。
- 核心方法 / 结论：论文提出 `GSA` 框架，用梯度而不是 loss 作为攻击特征，并通过 timestep subsampling 与 layer-wise aggregation 降低维度和成本。作者给出 `GSA1` 与 `GSA2` 两个实例化方法，并报告在 DDPM 和 Imagen 上都能取得很高的成员推断精度；在 CIFAR-10 上，`GSA1/GSA2` 的 AUC 达到 `0.999`，显著高于同条件下的 loss-based 对照。
- 和 DiffAudit 的关系：它对 DiffAudit 的价值在于为 white-box 路线提供了强上界参考，同时也明确指出复现该路线所需的关键资产是 checkpoint、训练配置和样本级梯度接口。换言之，这篇论文既是 white-box 方向的重要文献支点，也把后续工程工作的真实阻塞项暴露得很清楚。
- 开源仓库：[py85252876/GSA](https://github.com/py85252876/GSA)
- 阅读报告：[White-box Membership Inference Attacks against Diffusion Models](https://www.feishu.cn/docx/UExMdcM1do2z8dxffbiccIK6nOf)

## 背景与上下文

### DiffAudit 产品需求文档（PRD）

- 文件：[diffaudit-product-requirements.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/context/diffaudit-product-requirements.pdf)
- 内容简介：这份材料讨论的不是单点算法改进，而是如何把扩散模型成员推断、隐私泄露识别和合规证据输出组织成一个完整产品。它关注的核心问题是：面对可能记忆训练图像的扩散模型，审计方如何在不同访问权限下稳定发现风险，并把结果转化为业务和法规可理解的交付物。
- 核心方法 / 结论：文档提出的产品方案由三类检测信号组成：白盒 `DDIM Inversion` 重建误差、灰盒 `Cross-Attention` 热力图溯源和黑盒 `Likelihood/Loss` 判别；同时配套风险仪表盘、可视化溯源界面、模型加载器与标准合规报告模块。它的核心贡献不是证明新方法优于现有论文，而是把已有研究信号重组为一套可执行的工程路线图和产品闭环。
- 和 DiffAudit 的关系：对 DiffAudit 而言，这份 PRD 的价值在于把仓库中已经存在的黑盒、灰盒、白盒研究主线，统一编排成产品需求、里程碑和交付语言。它适合作为团队后续 UI 原型、报告生成器和审计演示系统的需求基线，但不能替代正式实验论文或复现实证材料。
- 开源仓库：[DeliciousBuding/DiffAudit](https://github.com/DeliciousBuding/DiffAudit/)
- 阅读报告：[DiffAudit 产品需求文档（PRD）阅读报告](https://www.feishu.cn/docx/PxuMd5asFolFMXxqRCNcroaindb)

### DiffAudit 团队入门扫盲文档

- 文件：[diffaudit-team-onboarding.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/context/diffaudit-team-onboarding.pdf)
- 内容简介：这份文档面向第一次接触扩散模型隐私审计的新成员，核心任务是解释 DiffAudit 到底在研究什么。它把问题收束为“扩散模型会不会记住训练样本并因此暴露成员信息”，并进一步说明项目目标不是提升图像生成质量，而是构建一套能够识别、解释和展示隐私风险的审计系统。
- 核心方法 / 结论：文档给出的核心方法不是新攻击算法，而是一条白盒优先的最小闭环：选择一个模型，准备成员与非成员样本，对图像做 DDIM 反演与重建，计算重建误差，比较两组差异，并把结果整理成可视化证据或报告。它还系统整理了成员推断、记忆、白盒/灰盒/黑盒、潜空间、注意力热力图等术语，用于统一团队内部的基础语言。
- 和 DiffAudit 的关系：它对 DiffAudit 的意义在于统一项目叙事和启动顺序。文档明确要求团队先打穿一个可展示的审计闭环，再逐步扩展黑盒、灰盒与防御建议，这对新成员 onboarding、申报材料撰写和后续报告产出都有直接价值；同时，它也提醒团队当前仓库的黑盒优先现实与文档中的白盒优先设想之间仍需做明确对齐。
- 开源仓库：[DeliciousBuding/DiffAudit](https://github.com/DeliciousBuding/DiffAudit/)
- 阅读报告：[DiffAudit 团队入门扫盲文档](https://www.feishu.cn/docx/HIY8dWNPjooAgHxjSuMc5UmCnTh)

## 综述与归档

### Tracing the Roots: Leveraging Temporal Dynamics in Diffusion Trajectories for Origin Attribution

- 文件：[2025-neurips-tracing-the-roots-origin-attribution-diffusion-trajectories.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/survey/2025-neurips-tracing-the-roots-origin-attribution-diffusion-trajectories.pdf)
- 内容简介：这篇论文讨论扩散模型的来源判定问题，不再把任务局限为传统 member / non-member membership inference，而是要求同时区分训练成员、模型新生成样本和外部样本，并进一步考虑特定模型归因。作者认为，如果不显式建模 belonging 类别，就会把很多并非训练数据的模型生成样本错误解释成 member，从而误读扩散模型的隐私风险。
- 核心方法 / 结论：方法上，作者不再依赖某个单独时间步的 denoising loss 阈值，而是沿整条 diffusion trajectory 提取跨时间步特征，包括 $L_t$、$\|\nabla_x L_t\|_2^2$ 和 $\|\nabla_\theta L_t\|_2^2$，再用线性 probe 做分类。实验表明，这种全轨迹表征比局部 “Goldilocks zone” 思路更稳健，也能在统一的 origin attribution 任务中超过 naive model-blind baseline，并给出据作者所述首个直接面向 diffusion 的 white-box model attribution 结果。
- 和 DiffAudit 的关系：对 DiffAudit 来说，这篇论文的意义在于校正评估口径，而不只是提供一个新分数。它说明扩散审计若不控制分布偏移、不比较 model-blind baseline、也不把 belonging 纳入标签空间，就容易高估 MIA 的实际解释力；同时，它为白盒路线提供了一个 trajectory-level 的上界参考，能帮助项目区分黑盒、灰盒和白盒审计的能力边界。
- 开源仓库：暂未找到
- 阅读报告：[Tracing the Roots: Leveraging Temporal Dynamics in Diffusion Trajectories for Origin Attribution](https://www.feishu.cn/docx/LUVXdHtvWo7D1PxU0cLc7s5InNe)

### DP-DocLDM: Differentially Private Document Image Generation Using Latent Diffusion Models

- 文件：[2025-icdar-dp-docldm-private-document-image-generation-latent-diffusion.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/survey/2025-icdar-dp-docldm-private-document-image-generation-latent-diffusion.pdf)
- 内容简介：这篇论文研究如何在文档图像分类场景下，用满足差分隐私约束的合成文档替代真实私有训练数据。作者指出，直接对下游分类器施加 DP-SGD 往往带来较大性能损失，因此更可行的问题是先训练一个私有生成器，再把真实敏感文档替换为合成数据用于常规训练。
- 核心方法 / 结论：方法上，作者提出 DP-DocLDM：先在大规模公有文档数据上预训练 KL autoencoder 和 conditional latent diffusion model，再在私有 RVL-CDIP 与 Tobacco3482 上用 DPDM 或 DP-Promise 做差分隐私微调。模型同时使用类别条件与 OCR 提取的 layout mask，并比较 class-conditional 与 per-label 两类私有训练策略。实验显示，`Layout+Class Cond. + per-label` 最有效，且在小规模 Tobacco3482 上，相比直接 DP-SGD 能显著提升下游分类准确率。
- 和 DiffAudit 的关系：对 DiffAudit 来说，这篇论文更适合作为 defense / survey 侧的重要参照，而不是攻击路线本身。它说明扩散模型不仅可能泄露训练数据，也可以被用作私有数据替代器；同时，它为“直接私有训练”与“先私有生成再常规训练”的路线比较提供了一个具体、可引用的文档场景案例。
- 开源仓库：暂未找到
- 阅读报告：[DP-DocLDM: Differentially Private Document Image Generation using Latent Diffusion Models](https://www.feishu.cn/docx/X8sAdAg7HosD8Nx6aUocbBLVnic)

### Privacy-Preserving Low-Rank Adaptation Against Membership Inference Attacks for Latent Diffusion Models

- 文件：[2025-aaai-privacy-preserving-lora-membership-inference-latent-diffusion-models.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/survey/2025-aaai-privacy-preserving-lora-membership-inference-latent-diffusion-models.pdf)
- 内容简介：这篇论文关注 latent diffusion model 在 LoRA 微调场景下的成员推断风险。作者指出，虽然 LoRA 以参数高效著称，但一旦在私有图像数据上做适配，模型仍可能泄露足以判断某个样本是否属于训练集的成员信号，因此轻量微调并不天然等于隐私安全。
- 核心方法 / 结论：论文先提出 MP-LoRA，把适配损失和代理成员推断器的 MI gain 写成 min-max 目标；随后发现这种直接加和的设计会导致优化不稳定。为此作者提出 SMP-LoRA，把 MI gain 放入分母形成比值型目标，并从局部光滑性分析与多组实验中论证该目标比 MP-LoRA 更稳定，能够在多个数据集上把 ASR、AUC 和 TPR 压到接近随机水平，同时基本保持 LoRA 的生成质量。
- 和 DiffAudit 的关系：对 DiffAudit 来说，这篇论文的意义在于它提供了一个与 LoRA 微调直接相关的防御对照基线，也提醒团队在解读扩散隐私结果时必须同时看攻击指标和生成质量指标。它不直接扩展主线攻击能力，但非常适合支撑“轻量微调为何会泄露、以及训练时如何缓解”这一研究叙事。
- 开源仓库：暂未找到
- 阅读报告：[Privacy-Preserving Low-Rank Adaptation Against Membership Inference Attacks for Latent Diffusion Models](https://www.feishu.cn/docx/Iiz3dYcU1orgUYxAqBkc7ivRnbh)

### Dual-Model Defense: Safeguarding Diffusion Models from Membership Inference Attacks through Disjoint Data Splitting

- 文件：[2024-arxiv-dual-model-defense-diffusion-membership-inference-disjoint-data-splitting.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/survey/2024-arxiv-dual-model-defense-diffusion-membership-inference-disjoint-data-splitting.pdf)
- 内容简介：这篇论文讨论扩散模型的成员推断防御问题。作者认为，扩散模型之所以会泄露成员信息，核心原因在于模型对训练样本的过拟合使成员与非成员之间出现稳定的去噪误差间隙，因此攻击者能够在白盒或黑盒场景下利用这一差异完成成员判定。
- 核心方法 / 结论：方法上，论文提出两条基于互斥数据切分的路线。`DualMD` 将训练集切成两个不相交子集，分别训练两个模型，并在推理时交替使用它们进行去噪；`DistillMD` 则利用“未见过该样本”的教师模型输出作为软目标蒸馏学生模型。实验表明，`DistillMD` 对白盒 MIA 的缓解最明显，而在文本到图像黑盒场景中，prompt diversification 是防御是否成立的关键前提。
- 和 DiffAudit 的关系：对 DiffAudit 来说，这篇论文的重要性不在于扩展攻击能力，而在于提供防御侧的系统对照。它说明扩散隐私风险与过拟合、prompt overfitting、以及 memorization 之间存在直接联系，因此适合被纳入项目的 survey 叙事，用来解释为什么某些成员信号会出现，以及哪些训练级改造可能压低这些信号。
- 开源仓库：暂未找到
- 阅读报告：[Dual-Model Defense: Safeguarding Diffusion Models from Membership Inference Attacks Through Disjoint Data Splitting](https://www.feishu.cn/docx/IS5FdwDU6o1ROTxHrzncHF4mnHe)

### DIFFENCE: Fencing Membership Privacy With Diffusion Models

- 文件：[2025-ndss-diffence-fencing-membership-privacy-diffusion-models.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/survey/2025-ndss-diffence-fencing-membership-privacy-diffusion-models.pdf)
- 内容简介：这篇论文研究图像分类模型的成员推断防御，目标是在不改目标模型训练流程、也不篡改输出接口的情况下，削弱黑盒攻击者利用成员与非成员预测分布差异进行推断的能力。作者把问题定义为 pre-inference defense：在样本进入目标分类器前先做扩散式重建，让模型面对的是语义相近但细节被改写的新输入。
- 核心方法 / 结论：方法上，DIFFENCE 对每个输入先做 forward diffusion 与 reverse denoising，生成多个重建候选，再只保留预测标签与原图一致的候选，以避免准确率下降。若防御者有更强先验，就在这些候选中挑选能更好缩小成员与非成员 logit 分布差异的样本。论文报告该方法可在多个数据集和既有防御之上继续降低 attack accuracy、attack AUC 以及低 FPR 区间的攻击能力，同时基本保持分类准确率和置信校准。
- 和 DiffAudit 的关系：对 DiffAudit 来说，这篇论文的重要性在于它提供了一条与攻击视角互补的输入侧防御路线，并明确指向成员推断最常利用的统计差异究竟是什么。它不直接扩展 DiffAudit 的攻击面，但非常适合作为 survey 中“预测分布差异如何被主动压平”的代表性防御论文，也能为后续比较黑盒审计与防御效果提供统一参照。
- 开源仓库：暂未找到
- 阅读报告：[DIFFENCE: Fencing Membership Privacy With Diffusion Models](https://www.feishu.cn/docx/XchVdTnrBoXzc9xYqy1cj5LHnlc)

### Defending Diffusion Models Against Membership Inference Attacks via Higher-Order Langevin Dynamics

- 文件：[2025-arxiv-defending-diffusion-models-membership-inference-higher-order-langevin-dynamics.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/survey/2025-arxiv-defending-diffusion-models-membership-inference-higher-order-langevin-dynamics.pdf)
- 内容简介：这篇论文讨论扩散模型的成员推断防御问题，重点不是设计更强攻击，而是回答在 `PIA` 这类依赖 score 网络的灰盒攻击下，是否可以通过改变扩散动力学本身来降低成员泄露。作者把问题放在敏感训练数据场景中，认为扩散模型虽然相对更抗 MIA，但仍然不能默认安全。
- 核心方法 / 结论：论文提出使用 critically-damped higher-order Langevin dynamics (`HOLD++`) 取代普通扩散过程，在状态空间中引入速度、加速度等辅助变量，让外部随机性更早混入轨迹，从而破坏 `PIA` 所依赖的确定性近似。理论上，作者给出一个基于 R\'enyi differential privacy 的上界分析；实验上，在 Swiss Roll 和 LJ Speech 上观察到更高的模型阶数与更大的辅助方差通常会降低 AUROC，而在 LJ Speech 中 `n=2` 还多次同时优于 `n=1` 的 FID 与 AUROC。
- 和 DiffAudit 的关系：对 DiffAudit 而言，这篇论文更适合作为防御路线和机制分析的代表材料，而不是当前主线攻击基线。它的直接价值在于提供一个很清晰的研究命题：成员隐私的缓解不一定只能靠训练正则化，也可以通过改变扩散轨迹的可确定性来实现，这对未来构建攻击-防御对照叙事很有帮助。
- 开源仓库：暂未找到
- 阅读报告：[Defending Diffusion Models Against Membership Inference Attacks via Higher-Order Langevin Dynamics](https://www.feishu.cn/docx/WqGFd3AXWopMALxe95KcxsKgnWu)

### Inference Attacks Against Graph Generative Diffusion Models

- 文件：[2026-arxiv-inference-attacks-graph-generative-diffusion-models.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/survey/2026-arxiv-inference-attacks-graph-generative-diffusion-models.pdf)
- 内容简介：这篇论文研究图生成扩散模型的训练数据泄露问题，核心关注点是：当攻击者只能黑盒访问目标模型、只能拿到其生成图集合时，是否仍然能恢复训练图结构、估计训练集统计属性，或判断某个图是否属于训练成员。作者把这一问题放在模型市场和 MLaaS 场景下讨论，因此强调的是弱访问条件下的真实输出泄露，而不是依赖内部 logits 或梯度的强攻击。
- 核心方法 / 结论：方法上，论文统一提出三类攻击。图重构攻击通过 REGAL 对齐生成图并取重叠边恢复训练图结构；属性推断攻击直接在生成图上统计密度、平均度等量来近似训练集分布；成员推断攻击则训练 shadow model，并利用输入图与生成图之间、以及生成图彼此之间的结构相似度构造攻击特征。实验表明，图重构 F1 最高可达 0.99，成员推断在同分布设定下 AUC 最高可达 0.999，跨数据集设定下仍有 0.895，同时作者还提出了两类基于最不重要边翻转的防御。
- 和 DiffAudit 的关系：对 DiffAudit 来说，这篇论文的意义主要体现在 black-box 叙事层面：它证明了“只暴露最终生成样本集合”并不等于安全，训练数据的结构和统计信息仍可能从输出分布中被反推出去。不过论文研究对象是图生成扩散模型，很多具体特征与评估量都带有图域专属性，因此它更适合作为结构化生成模型的 survey 参照，而不是当前图像扩散审计路线的直接实现模板。
- 开源仓库：暂未找到
- 阅读报告：[Inference Attacks Against Graph Generative Diffusion Models](https://www.feishu.cn/docx/XGnxdoo1zoC3C5xxPuCcjaVJnpd)

### Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models

- 文件：[2025-arxiv-perturb-a-model-not-an-image-anti-personalized-diffusion-models.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/survey/2025-arxiv-perturb-a-model-not-an-image-anti-personalized-diffusion-models.pdf)
- 内容简介：这篇论文讨论的是个性化扩散模型的隐私保护问题。作者认为，现有基于图像投毒的防护方法依赖不现实的前提，例如要求用户控制全部相关图像，而且在攻击者混入少量干净图像或施加简单变换时容易失效，因此难以满足真实平台侧的防护需求。
- 核心方法 / 结论：论文提出 Anti-Personalized Diffusion Models（APDM），把防护对象从输入图像转到模型参数。其核心由 Direct Protective Optimization（DPO）和 Learning to Protect（L2P）组成：前者通过正负样本对直接压制目标主体的个性化能力，后者通过模拟未来个性化轨迹来累积保护梯度。实验表明，APDM 在 `person` 和 `dog` 等主体上显著优于 AdvDM、Anti-DreamBooth、SimAC、PAP 等图像投毒基线，同时只带来有限的通用生成能力退化。
- 和 DiffAudit 的关系：对 DiffAudit 而言，这篇论文的重要性在于它代表了一条与成员推断不同但强相关的“模型级防护”路线。它说明当扩散模型已经具备可被滥用的个性化能力时，服务提供方可以直接通过参数更新部署针对特定主体的保护机制，这为项目后续整合攻击分析与可部署防护叙事提供了高价值参考。
- 开源仓库：暂未找到
- 阅读报告：[Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models](https://www.feishu.cn/docx/KVaAdUuNkotcj9xi2L1cRlh8nod)

### Diffusion Privacy Literature Survey Index

- 文件：[survey-index-diffusion-privacy-literature.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/survey/survey-index-diffusion-privacy-literature.pdf)
- 内容简介：这份材料处理的问题不是提出新的扩散模型隐私防护算法，而是整理当前与扩散模型成员推断防御相关的代表性文献。它把多篇论文压缩为统一格式的索引条目，记录方法名称、模型类型、防御思路、防御类别、优缺点以及是否可迁移到扩散模型。
- 核心方法 / 结论：材料的核心贡献是建立了一个便于浏览的防御路线索引，而不是给出新的实验结论。当前版本共列出 7 条路线，覆盖差分隐私训练、LoRA 微调防御、双模型拆分与蒸馏、推理前扰动、朗之万动力学扰动、图扩散显著性扰动和 anti-personalized diffusion models，并对每条路线给出简要迁移判断。
- 和 DiffAudit 的关系：对 DiffAudit 来说，这份材料的价值在于它可以作为 survey 轨道的快速入口，帮助团队确定哪些防御工作值得逐篇深读和后续复现。与此同时，它只是二级文献索引，不能替代原论文证据，因此适合做路线导航，不适合直接作为最终技术结论来源。
- 开源仓库：不适用
- 阅读报告：[扩散模型隐私防护文献索引整理材料阅读报告](https://www.feishu.cn/docx/RPMyd7FbfoFnM9xHMtBcf2eKn5d)
