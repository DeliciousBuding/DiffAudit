# 论文索引

这份文档面向团队内部阅读与飞书同步，目标是用统一口径说明每篇材料在 DiffAudit 课题中的定位、价值和当前可用性。

字段说明：

- `文件`：仓库内 PDF 的规范化路径
- `内容简介`：说明这篇材料关注的问题、威胁模型或使用场景
- `核心方法 / 结论`：概括论文采用的主要思路、技术路线或实验结论
- `和 DiffAudit 的关系`：说明这篇材料为什么值得进入当前索引
- `开源仓库`：优先给官方或作者公开实现；找不到就明确写“暂未找到”

## 黑盒

### Towards Black-Box Membership Inference Attack for Diffusion Models

- 文件：[2024-arxiv-towards-black-box-membership-inference-diffusion-models.pdf](black-box/2024-arxiv-towards-black-box-membership-inference-diffusion-models.pdf)
- 内容简介：这篇论文讨论最严格的黑盒成员推断场景。攻击者无法访问模型参数、训练损失或中间特征，只能通过图像变体接口反复调用服务，判断某张候选图像是否出现在训练集中。
- 核心方法 / 结论：作者提出 `REDIFFUSE`，通过多次生成图像变体，再比较原图与变体之间的相似性分布来构造成员分数。论文结论说明，即使只开放变体 API，训练成员与非成员之间仍可能残留可利用的行为差异。
- 和 DiffAudit 的关系：它是最贴近“外部 API 审计”场景的黑盒基线，对应仓库里的 `variation` 路线，也是团队讨论真实在线服务审计时最常引用的参考论文之一。
- 开源仓库：暂未找到

### Membership Inference on Text-to-Image Diffusion Models via Conditional Likelihood Discrepancy

- 文件：[2024-neurips-clid-membership-inference-text-to-image-diffusion.pdf](black-box/2024-neurips-clid-membership-inference-text-to-image-diffusion.pdf)
- 内容简介：这篇工作聚焦文生图扩散模型，研究对象不再是“图像边缘分布是否被记住”，而是“给定文本条件后，模型是否对训练样本形成异常偏好”。论文的问题设置更接近真实文生图服务。
- 核心方法 / 结论：作者提出 `CLiD`，利用条件似然差异来衡量成员与非成员在文本条件下的生成一致性偏差。论文表明，文本条件会放大过拟合信号，因此成员推断在 text-to-image 场景中可以比传统无条件设置更有效。
- 和 DiffAudit 的关系：它是当前最重要的文生图黑盒论文之一，对 caption 条件明确存在的场景尤其关键，也是仓库 `clid` 路线的直接理论来源。
- 开源仓库：[zhaisf/CLiD](https://github.com/zhaisf/CLiD)

### Black-box Membership Inference Attacks against Fine-tuned Diffusion Models

- 文件：[2025-ndss-black-box-membership-inference-fine-tuned-diffusion-models.pdf](black-box/2025-ndss-black-box-membership-inference-fine-tuned-diffusion-models.pdf)
- 内容简介：这篇论文面向现实里更常见的“预训练扩散模型经过下游微调”场景，研究如何在黑盒条件下判断某张图像是否属于微调训练集。它特别关注攻击者只有输出结果而无法直接读取模型内部状态时的审计能力。
- 核心方法 / 结论：论文提出了一套以重建误差和生成行为差异为核心的系统攻击框架，并在微调模型上取得了很强的区分效果。它说明微调过程会留下可被黑盒审计利用的成员信号，而且这些信号可以通过统一流程工程化复现。
- 和 DiffAudit 的关系：这是当前 DiffAudit 黑盒主线最核心的目标论文，对应仓库里的 `recon` 路线，也是当前最接近“先拿真实资产、再产出审计证据”的主执行方向。
- 开源仓库：[py85252876/Reconstruction-based-Attack](https://github.com/py85252876/Reconstruction-based-Attack)

### Membership Inference Attacks for Face Images Against Fine-tuned Latent Diffusion Models

- 文件：[2025-visapp-membership-inference-face-fine-tuned-latent-diffusion-models.pdf](black-box/2025-visapp-membership-inference-face-fine-tuned-latent-diffusion-models.pdf)
- 内容简介：这篇论文把成员推断问题收缩到人脸图像这一更敏感、也更接近现实合规诉求的场景。核心问题是：在针对人脸数据微调的 latent diffusion 模型上，攻击者能否推断某批面孔是否被用于训练。
- 核心方法 / 结论：作者分析了 watermark、guidance scale、prompt 等因素对攻击成功率的影响，说明具体生成条件会显著改变成员信号的强弱。论文的主要价值不在于提出全新框架，而在于补充了一个高度敏感、可落地的风险样例。
- 和 DiffAudit 的关系：它展示了“面向敏感数据的人脸隐私审计”这一具体落地场景，适合在团队后续讨论行业案例、合规叙事和面向客户的审计报告时作为补充材料。
- 开源仓库：[osquera/MIA_SD](https://github.com/osquera/MIA_SD)

## 灰盒

### Are Diffusion Models Vulnerable to Membership Inference Attacks?

- 文件：[2023-icml-secmi-membership-inference-diffusion-models.pdf](gray-box/2023-icml-secmi-membership-inference-diffusion-models.pdf)
- 内容简介：这是较早系统研究扩散模型成员推断的代表作之一，关注攻击者能够部分接触扩散过程信息的灰盒场景。论文试图回答的核心问题是：扩散模型内部的逐步去噪轨迹，是否天然泄露成员信息。
- 核心方法 / 结论：作者提出 `SecMI`，通过比较扩散过程中的后验估计误差来判断成员关系。论文结果表明，扩散时间步上的误差模式本身就包含稳定的成员信号，因此扩散模型并不像直觉上那样“天然更难被做 MIA”。
- 和 DiffAudit 的关系：它是当前灰盒路线最基础的基线之一，对应仓库里的 `secmi` 实现，也是团队理解“扩散轨迹如何泄露隐私”的入门论文。
- 开源仓库：[jinhaoduan/SecMI](https://github.com/jinhaoduan/SecMI)

### SIDE: Surrogate Conditional Data Extraction from Diffusion Models

- 文件：[2024-arxiv-side-extracting-training-data-unconditional-diffusion-models.pdf](gray-box/2024-arxiv-side-extracting-training-data-unconditional-diffusion-models.pdf)
- 内容简介：这篇工作关注的不只是“样本是否属于训练集”，而是更强的“训练数据能否被抽取出来”。论文研究的是无条件扩散模型在被赋予代理条件后，是否会暴露出更直接的数据泄露风险。
- 核心方法 / 结论：作者提出代理条件构造方式，将原本无条件的扩散模型变成可以被引导的数据提取对象。论文传达的重点是，成员推断只是更大隐私风险谱系中的一个层级，扩散模型还可能面临更直接的数据恢复问题。
- 和 DiffAudit 的关系：它提醒团队审计视角不能只盯着 MIA 指标，还要关注更强的数据提取与训练样本恢复风险，适合作为灰盒到白盒风险升级的过渡材料。
- 开源仓库：暂未找到

### Unveiling Structural Memorization: Structural Membership Inference Attack for Text-to-Image Diffusion Models

- 文件：[2024-arxiv-structural-memorization-membership-inference-text-to-image-diffusion.pdf](gray-box/2024-arxiv-structural-memorization-membership-inference-text-to-image-diffusion.pdf)
- 内容简介：这篇论文关注文生图模型中的“结构性记忆”，核心观点是模型不一定逐像素记住训练样本，但可能对布局、构图和语义结构形成稳定记忆。论文因此把成员推断的关注点从像素级相似扩展到了结构级信号。
- 核心方法 / 结论：作者围绕“结构保持”构造攻击特征，并证明结构层面的相似性能够提供额外的成员证据。论文给出的信息是，审计证据不必局限于重建误差，结构信息也可能是有效攻击面。
- 和 DiffAudit 的关系：它为团队提供了不同于 `recon` 的证据思路，说明未来灰盒或黑盒审计都可以尝试引入更中层的结构信号，而不仅是像素或 loss 层面的统计量。
- 开源仓库：暂未找到

### An Efficient Membership Inference Attack for the Diffusion Model by Proximal Initialization

- 文件：[2024-iclr-pia-proximal-initialization.pdf](gray-box/2024-iclr-pia-proximal-initialization.pdf)
- 内容简介：这篇论文研究如何在扩散模型上更高效地做成员推断，重点解决“攻击成本高、查询步数多”的问题。它不仅覆盖图像，也讨论了攻击思想向 TTS 等非视觉任务的延展。
- 核心方法 / 结论：作者提出 `PIA`，通过更接近真实扩散轨迹的初始化方式，减少无效搜索并强化成员与非成员之间的区分度。论文结果说明，好的初始化策略本身就能显著提升攻击效率，不必完全依赖更复杂的特征工程。
- 和 DiffAudit 的关系：它是当前灰盒实现主线之一，对应仓库里的 `pia` 路线，也为团队后续做“更便宜的审计流程”提供了参考。
- 开源仓库：[kong13661/PIA](https://github.com/kong13661/PIA)

### Score-based Membership Inference on Diffusion Models

- 文件：[2025-arxiv-sima-score-based-membership-inference-diffusion-models.pdf](gray-box/2025-arxiv-sima-score-based-membership-inference-diffusion-models.pdf)
- 内容简介：这篇论文把关注点放在模型预测的 score / noise 范数上，试图回答“是否只看一次预测结果就足以完成成员推断”。它强调的是更轻量、更直接的灰盒攻击设置。
- 核心方法 / 结论：作者提出 `SimA`，并从理论与实验两方面说明 score 范数本身就可能携带成员信息。论文的意义在于证明，成员信号未必需要完整轨迹或复杂后处理，有时单次查询的统计量就足够形成有效判别。
- 和 DiffAudit 的关系：它适合作为未来灰盒基准中的高效 score-based 参照方法，也有助于团队比较 `SecMI`、`PIA` 与更轻量方案之间的工程成本差异。
- 开源仓库：[mx-ethan-rao/SimA](https://github.com/mx-ethan-rao/SimA)

### Noise Aggregation Analysis Driven by Small-Noise Injection: Efficient Membership Inference for Diffusion Models

- 文件：[2025-arxiv-small-noise-injection-membership-inference-diffusion-models.pdf](gray-box/2025-arxiv-small-noise-injection-membership-inference-diffusion-models.pdf)
- 内容简介：这篇工作研究在输入端注入小幅噪声后，成员样本与非成员样本在预测稳定性上的差异。论文关注的是一种低干预、低额外成本的灰盒检测路径。
- 核心方法 / 结论：方法核心是观察小噪声注入后预测噪声的聚合程度，成员样本通常会表现出更稳定的响应模式。论文说明，轻微扰动就可能放大隐藏的成员信号，因此这是一条兼顾效率与效果的候选路线。
- 和 DiffAudit 的关系：这是一个查询成本较低、便于做批量实验的候选方法，适合团队后续在灰盒线路里扩展“轻量扰动型”基准。
- 开源仓库：暂未找到

### CDI: Copyrighted Data Identification in Diffusion Models

- 文件：[2025-cvpr-cdi-copyrighted-data-identification-diffusion-models.pdf](gray-box/2025-cvpr-cdi-copyrighted-data-identification-diffusion-models.pdf)
- 内容简介：这篇论文把问题从“单张图片是不是训练成员”提升到了“某个版权数据集是否被模型使用”。它讨论的是更贴近现实合规与举证场景的数据集级识别问题。
- 核心方法 / 结论：作者通过聚合样本级信号和统计检验，构造更适合形成外部举证的数据集级证据。论文传达的关键结论是：对于合规审计和版权争议来说，数据集级判断往往比单样本判断更有实际价值。
- 和 DiffAudit 的关系：这条路线和未来产品化、报告化非常契合，因为企业侧真正需要的常常不是“某一张图像是否被记住”，而是“某个数据集是否被使用”。
- 开源仓库：[sprintml/copyrighted_data_identification](https://github.com/sprintml/copyrighted_data_identification)

### Noise as a Probe: Membership Inference Attacks on Diffusion Models Leveraging Initial Noise

- 文件：[2026-arxiv-noise-as-a-probe-membership-inference-diffusion-models.pdf](gray-box/2026-arxiv-noise-as-a-probe-membership-inference-diffusion-models.pdf)
- 内容简介：这篇论文把初始噪声本身视为攻击探针，重点研究小数据集微调模型在初始噪声层面是否残留语义信息。问题设置比传统灰盒更强调“攻击入口的可控性”。
- 核心方法 / 结论：作者利用扩散初始噪声中的残留语义构造成员推断信号，说明模型泄露不一定只能从中间层或完整轨迹中观测。论文结论表明，初始噪声空间也可能成为有效的审计接口。
- 和 DiffAudit 的关系：它提供了不依赖显式中间层输出的另一条灰盒路径，对团队后续思考“接口越少时还能拿什么信号”很有参考价值。
- 开源仓库：暂未找到

### No Caption, No Problem: Caption-Free Membership Inference via Model-Fitted Embeddings

- 文件：[2026-openreview-mofit-caption-free-membership-inference.pdf](gray-box/2026-openreview-mofit-caption-free-membership-inference.pdf)
- 内容简介：这篇论文正面解决真实审计里经常遇到的一个问题：手里只有图像，没有训练 caption。它研究如何在缺失文本条件的情况下，依然对 text-to-image 模型开展成员推断。
- 核心方法 / 结论：作者提出 model-fitted embedding，用模型自身拟合出的嵌入来替代不可获得的原始 caption。论文说明，只要能构造足够贴近模型内部对齐方式的代理表示，就可以在 caption 缺失时保留可用的成员信号。
- 和 DiffAudit 的关系：这篇论文非常实用，因为很多真实审计场景里，团队拿到的往往只有图像样本而没有训练元数据；它为未来黑盒与灰盒结合的审计流程提供了现实可行的补位方案。
- 开源仓库：[JoonsungJeon/MoFit](https://github.com/JoonsungJeon/MoFit)

## 白盒

### Finding NeMo: Localizing Neurons Responsible for Memorization in Diffusion Models

- 文件：[2024-neurips-finding-nemo-localizing-memorization-neurons-diffusion-models.pdf](white-box/2024-neurips-finding-nemo-localizing-memorization-neurons-diffusion-models.pdf)
- 内容简介：这篇工作不再满足于判断“模型有没有记忆训练样本”，而是进一步追问“到底是哪些神经元在承载这种记忆”。它关注的是扩散模型中的白盒定位与解释问题。
- 核心方法 / 结论：作者提出 `NEMO`，试图定位与记忆泄露最相关的神经元，并展示停用这些神经元后可以降低泄露风险。论文表明，成员记忆并非完全分散在全模型中，而是可能集中体现在局部神经元子集上。
- 和 DiffAudit 的关系：它是白盒解释和白盒缓解路线的重要参考。团队未来如果要把产品能力从“检测”扩展到“定位与缓解”，这篇论文是绕不过去的基石材料。
- 开源仓库：[ml-research/localizing_memorization_in_diffusion_models](https://github.com/ml-research/localizing_memorization_in_diffusion_models)

### White-box Membership Inference Attacks against Diffusion Models

- 文件：[2025-local-mirror-white-box-membership-inference-diffusion-models.pdf](white-box/2025-local-mirror-white-box-membership-inference-diffusion-models.pdf)
- 内容简介：这是同一白盒论文的本地镜像版本，研究对象是攻击者可访问更丰富模型内部信息时，哪些特征最适合做扩散模型成员推断。它主要服务于技术对照与材料留档。
- 核心方法 / 结论：论文系统比较了多类白盒特征，尤其强调梯度信息相较于单纯 loss 更有判别力。镜像版的作用主要是方便本地查阅与交叉核对，不应作为对外引用的首选版本。
- 和 DiffAudit 的关系：它是未来白盒攻击引擎的重要技术来源之一，但在团队正式对外沟通和文档引用时，建议以后面的 PoPETs 正式版作为 canonical copy。
- 开源仓库：[py85252876/GSA](https://github.com/py85252876/GSA)

### White-box Membership Inference Attacks against Diffusion Models

- 文件：[2025-popets-white-box-membership-inference-diffusion-models.pdf](white-box/2025-popets-white-box-membership-inference-diffusion-models.pdf)
- 内容简介：这是上述白盒工作的正式 PoPETs 版本，也是当前更适合团队统一引用的正式文献。论文系统梳理了扩散模型白盒成员推断中的有效特征和攻击条件。
- 核心方法 / 结论：作者比较了梯度、loss 等多种白盒特征，并得出梯度特征在白盒场景下更稳定、更具区分力的结论。论文因此把白盒攻击的重点从“简单统计量”转向“训练动态和梯度结构”。
- 和 DiffAudit 的关系：如果后续白盒方向只保留一篇主论文，这篇应作为 canonical copy。它为团队未来建设白盒攻击与解释模块提供了最直接的理论与实验依据。
- 开源仓库：[py85252876/GSA](https://github.com/py85252876/GSA)

## 背景与上下文

### DiffAudit 产品需求文档（PRD）

- 文件：[diffaudit-product-requirements.pdf](context/diffaudit-product-requirements.pdf)
- 内容简介：这不是论文，而是 DiffAudit 的产品需求文档。它说明项目为什么不能停留在“复现单篇论文”，而要进一步走向可交付的审计系统与可展示的输出形态。
- 核心方法 / 结论：文档明确了白盒、灰盒、黑盒、报告生成等模块边界，也定义了团队为何要从研究复现转向工程化审计产品。它的作用是统一研发、研究和展示层面的目标。
- 和 DiffAudit 的关系：这是系统定位和产品边界的内部依据，也是飞书首页状态页应该对齐的上层背景。
- 开源仓库：[DeliciousBuding/DiffAudit](https://github.com/DeliciousBuding/DiffAudit/)

### DiffAudit 团队入门扫盲文档

- 文件：[diffaudit-team-onboarding.pdf](context/diffaudit-team-onboarding.pdf)
- 内容简介：这份文档面向新成员，用更通俗的方式解释“扩散模型隐私审计”到底在做什么、为什么值得做，以及团队内部使用的基础术语。
- 核心方法 / 结论：它不提供新算法，而是通过统一概念、威胁模型和项目边界，降低不同成员进入课题时的理解成本。对跨职能协作来说，这类材料的重要性不低于单篇算法论文。
- 和 DiffAudit 的关系：这是内部共识和团队 onboarding 材料，适合作为飞书文档读者进入算法索引前的背景补充。
- 开源仓库：[DeliciousBuding/DiffAudit](https://github.com/DeliciousBuding/DiffAudit/)

## 综述与归档

### Tracing the Roots: Leveraging Temporal Dynamics in Diffusion Trajectories for Origin Attribution

- 文件：[2025-neurips-tracing-the-roots-origin-attribution-diffusion-trajectories.pdf](survey/2025-neurips-tracing-the-roots-origin-attribution-diffusion-trajectories.pdf)
- 内容简介：这篇工作关注更广义的来源归因问题，不只区分成员与非成员，还尝试判断样本究竟来自训练集、模型生成还是外部来源。它把成员推断放到了更大的溯源框架下讨论。
- 核心方法 / 结论：作者利用整条 diffusion trajectory 的时间动态来做来源识别，说明轨迹级信息不仅能服务于 MIA，也可以支撑更细粒度的 origin attribution。论文因此扩大了团队理解“扩散隐私审计”边界的视角。
- 和 DiffAudit 的关系：它对“成员推断只是更广义溯源问题的一部分”这个视角很有帮助，适合团队做长期路线设计或报告中的背景拓展。
- 开源仓库：暂未找到

### DP-DocLDM: Differentially Private Document Image Generation Using Latent Diffusion Models

- 文件：[2025-icdar-dp-docldm-private-document-image-generation-latent-diffusion.pdf](survey/2025-icdar-dp-docldm-private-document-image-generation-latent-diffusion.pdf)
- 内容简介：这篇论文将差分隐私引入文档图像生成场景，关注如何在 latent diffusion 框架下生成既可用又尽量不泄露训练隐私的文档图像数据。
- 核心方法 / 结论：作者尝试把差分隐私训练或采样思想与文档生成任务结合，说明防御路线并不局限于自然图像。论文的价值在于把扩散隐私问题延伸到了文档图像这一更专业的数据形态。
- 和 DiffAudit 的关系：它属于防御与隐私保护方向的参考材料，适合作为团队理解“攻击之外还有哪些缓解路线”的补充。
- 开源仓库：暂未找到

### Privacy-Preserving Low-Rank Adaptation Against Membership Inference Attacks for Latent Diffusion Models

- 文件：[2025-aaai-privacy-preserving-lora-membership-inference-latent-diffusion-models.pdf](survey/2025-aaai-privacy-preserving-lora-membership-inference-latent-diffusion-models.pdf)
- 内容简介：这篇工作关注 LoRA 微调在 latent diffusion 上带来的成员推断风险，核心问题是如何在保持轻量微调优势的同时抑制隐私泄露。
- 核心方法 / 结论：作者提出在低秩适配过程中加入隐私保护设计，使模型在保持一定任务性能的同时降低被做成员推断的风险。论文表明，LoRA 这类工程上非常实用的微调方式，也需要被纳入隐私审计讨论。
- 和 DiffAudit 的关系：它更偏“怎么防”而不是“怎么攻”，适合作为团队在主线攻击工作之外整理防御对照材料时的重要参考。
- 开源仓库：暂未找到

### Dual-Model Defense: Safeguarding Diffusion Models from Membership Inference Attacks through Disjoint Data Splitting

- 文件：[2024-arxiv-dual-model-defense-diffusion-membership-inference-disjoint-data-splitting.pdf](survey/2024-arxiv-dual-model-defense-diffusion-membership-inference-disjoint-data-splitting.pdf)
- 内容简介：这篇论文研究如何通过双模型训练和数据拆分来降低扩散模型的成员推断风险，关注的是更偏部署侧的防御方案。
- 核心方法 / 结论：作者通过 disjoint data splitting、私有推理和蒸馏等设计，削弱成员与非成员之间可被攻击利用的差异。论文说明，防御不仅可以在训练算法层面做，也可以在系统架构层面做。
- 和 DiffAudit 的关系：它适合作为防御路线的归档材料，不是当前主线攻击基线，但对团队后续构建“检测后如何减缓”的完整叙事有帮助。
- 开源仓库：暂未找到

### DIFFENCE: Fencing Membership Privacy With Diffusion Models

- 文件：[2025-ndss-diffence-fencing-membership-privacy-diffusion-models.pdf](survey/2025-ndss-diffence-fencing-membership-privacy-diffusion-models.pdf)
- 内容简介：这篇工作讨论一种推理前防御思路，即先对输入进行扩散式重生成，再交给目标模型处理，以此削弱成员与非成员之间的可区分差异。
- 核心方法 / 结论：论文提出把扩散模型作为隐私防护层使用，让攻击者难以直接利用原始输入中的成员信号。它的主要价值在于展示了“扩散模型本身也可以拿来做防御”这一思路。
- 和 DiffAudit 的关系：它是扩散模型防御路线里较有代表性的一篇论文，适合团队在未来产品报告中补充“防御与缓解”章节。
- 开源仓库：暂未找到

### Defending Diffusion Models Against Membership Inference Attacks via Higher-Order Langevin Dynamics

- 文件：[2025-arxiv-defending-diffusion-models-membership-inference-higher-order-langevin-dynamics.pdf](survey/2025-arxiv-defending-diffusion-models-membership-inference-higher-order-langevin-dynamics.pdf)
- 内容简介：这篇论文从采样与动力学角度讨论扩散模型的隐私防御，研究如何利用更高阶的 Langevin 动力学在生成过程中引入额外随机性。
- 核心方法 / 结论：作者通过调整采样动力学来减弱成员与非成员之间可被观察到的稳定差异。论文更偏理论和机制分析，说明防御未必只能依赖训练时正则化，也可以从采样过程本身入手。
- 和 DiffAudit 的关系：它提供了另一种偏理论化的防御思路，适合团队在整理研究版图时作为“采样过程防御”分支的代表。
- 开源仓库：暂未找到

### Inference Attacks Against Graph Generative Diffusion Models

- 文件：[2026-arxiv-inference-attacks-graph-generative-diffusion-models.pdf](survey/2026-arxiv-inference-attacks-graph-generative-diffusion-models.pdf)
- 内容简介：这篇工作把隐私攻击讨论扩展到了图生成扩散模型，说明扩散隐私问题并不局限于自然图像或文生图场景。
- 核心方法 / 结论：论文研究图结构数据上的成员或属性泄露风险，提示扩散模型的隐私问题具有跨模态、跨数据结构的共性。它的意义主要在于拓展边界，而不是直接服务当前图像主线。
- 和 DiffAudit 的关系：它有助于团队理解未来跨模态、跨数据结构扩展的可能方向，是路线扫描中的前瞻性材料。
- 开源仓库：暂未找到

### Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models

- 文件：[2025-arxiv-perturb-a-model-not-an-image-anti-personalized-diffusion-models.pdf](survey/2025-arxiv-perturb-a-model-not-an-image-anti-personalized-diffusion-models.pdf)
- 内容简介：这篇论文不再试图保护输入图像本身，而是从模型侧入手，减少模型被用于个性化生成或针对特定主体进行定制化复现的能力。
- 核心方法 / 结论：作者提出“anti-personalized diffusion model”的防护视角，强调通过扰动模型行为而不是扰动输入图像来提升隐私鲁棒性。它代表的是成员推断之外、更广义的生成隐私保护方向。
- 和 DiffAudit 的关系：它更偏“反个性化/隐私保护”方向，是成员推断之外的重要补充，也是当前本地附件里唯一因文件过大而暂不上传飞书的论文。
- 开源仓库：暂未找到

### Diffusion Privacy Literature Survey Index

- 文件：[survey-index-diffusion-privacy-literature.pdf](survey/survey-index-diffusion-privacy-literature.pdf)
- 内容简介：这是本地整理的归档索引文件，用来记录补充性的综述与参考材料，不对应某一篇独立研究论文。
- 核心方法 / 结论：它的作用主要是帮助团队在扩散隐私方向维护长期材料清单、补充阅读入口和分类信息，而不是提供单一技术结论。
- 和 DiffAudit 的关系：主要承担维护和整理功能，适合作为团队继续扩展 survey 区域时的辅助入口。
- 开源仓库：不适用
