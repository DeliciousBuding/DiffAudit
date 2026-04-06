#### 相关文献整理

#### 格式:

【方法名称 / 论文标题

发表时间与会议/期刊(NeurIPS / ICML / CCS / USENIX 等) 针对哪类模型(分类模型 / 生成模型 / 扩散模型)

防御核心思路(一两句话)

防御类别(CCFA 类 / 差分隐私类 / 正则化类 / 机器遗忘类…) 优缺点 / 局限性

是否可迁移到扩散模型】

# 一、DP-DocLDM / \*DP-DocLDM: Differentially Private Document Image Generation using Latent Diffusion Models\*

2025 年,arXiv 预印本(后续投 ICDAR 2025 会议)

潜在扩散模型(LDM,生成模型/扩散模型,专门针对文档图像生成 场景)

防御核心思路:针对文档图像生成场景,将差分隐私随机梯度下降 (DP-SGD)与连续时间扩散模型结合,在训练阶段为潜在扩散模型提 供严格的差分隐私保障,从根源阻断成员推理攻击等隐私泄露风险, 同时生成可用于下游任务的合规文档图像。

#### 差分隐私类

优点:提供可量化的严格差分隐私保障,是隐私保护生成模型的合 规基准方法;针对文档图像场景做了定制化优化,可生成满足隐私要 求的合成数据用于下游任务。

缺点:会显著降低生成样本的质量(FID 指标大幅上升),存在严 重的隐私-效用权衡;计算开销大,训练成本高;仅针对文档图像场 景,通用性有限。。

原生针对扩散模型设计,完全适配。。

## 二、MP-LoRA / SMP-LoRA /Privacy-Preserving Low-Rank Adaptation Against Membership Inference Attacks for Latent Diffusion Models(MP-LoRA)

2025 年,AAAI(人工智能顶会)

潜扩散模型(LDM,生成模型)

 防御核心思路:针对 LoRA 微调的潜扩散模型,提出最小-最大优化 框架,训练代理攻击模型最大化成员推断收益,同时让扩散模型最小 化攻击收益与微调损失的和,在微调阶段防御成员推断。

对抗正则化类

优点:仅微调 LoRA 参数,计算开销远低于全模型训练;在隐私保 护和生成质量间取得较好平衡,不破坏模型个性化能力;

缺点:仅针对 LoRA 微调场景,对原生训练的扩散模型适配性有限; 对强成员推断攻击的防御能力弱于差分隐私方法。

原生针对潜扩散模型设计,可迁移到全参数训练的扩散模型。。

## 三、DualMD / DistillIMD / DUAL-MODEL DEFENSE: SAFEGUARDING DIFFUSION MODELS FROM MEMBERSHIP INFERENCE ATTACKS THROUGH DISJOINT DATA SPLITTING(DualMD / DistillMD)

2024 年,arXiv 预印本(已投顶会)

扩散模型(生成模型)

防御核心思路:将原始数据集拆分为两个不相交子集,分别训练两 个独立扩散模型;DualMD 通过双模型推理 pipeline 防御黑盒攻击, DistillMD 通过知识蒸馏同时防御黑盒/白盒攻击,从数据拆分层面 降低过拟合带来的隐私泄露。

数据拆分+蒸馏正则化类

优点:无需修改模型结构,推理阶段即可部署(DualMD);DistillMD 可同时防御黑盒/白盒攻击,对生成质量影响小;

缺点:数据集拆分后单模型训练数据减少,可能影响模型泛化能力; DistillMD 需要额外蒸馏训练,增加计算开销。

原生针对扩散模型设计,完全适配。。

# 四 、 DIFFENCE/DIFFENCE: Fencing Membership Privacy With Diffusion Models

 2025 年,NDSS(网络与分布式系统安全顶会,安全四大顶会之一) 扩散模型(生成模型)

防御核心思路:提出推理前预处理的全新防御范式,在不修改模型、 不重训练的前提下,通过预处理输入数据混淆成员/非成员的特征差 异,即插即用,可与所有现有防御方法叠加使用。

推理阶段扰动类(CCFA/对抗扰动类)

优点:无需重训练,零成本部署;不损失模型生成质量;可与差分 隐私、正则化等方法叠加,增强防御效果;

缺点:对强白盒成员推断攻击的防御能力有限,更适合黑盒场景; 需要针对不同模型调整预处理策略。

原生针对扩散模型设计,完全适配,可迁移到其他生成模型。。

五、高阶朗之万动力学防御(Higher-Order Langevin Dynamics)

## /DEFENDING DIFFUSION MODELS AGAINST MEMBERSHIP INFERENCE ATTACKS VIA HIGHER-ORDER LANGEVIN DYNAMICS

2025 年,arXiv 预印本

扩散模型(生成模型)

防御核心思路:引入临界阻尼高阶朗之万动力学,在扩散过程中加 入辅助变量和外部随机性,在推理阶段破坏敏感数据的特征可区分 性,降低成员推断的准确性。

推理阶段动力学扰动类

优点:理论上可证明隐私增强效果;对生成质量影响小;

缺点:仅在语音、小数据集上验证,图像扩散模型的泛化性待验证; 推理阶段增加计算开销。

原生针对扩散模型设计,完全适配。。

#### 六、Saliency Map Perturbation / Inference Attacks Against Graph Generative Diffusion Models

USENIX Security 2026 图扩散模型隐私泄露量化研究与防御 (Saliency Map Perturbation)

2026 年,USENIX Security(安全四大顶会之一)

图扩散模型(生成模型)

防御核心思路:不使用效用损失大的差分隐私,而是计算图中每条 边对模型损失的梯度贡献(显著性图),对高贡献边施加针对性扰动, 在不破坏图结构的前提下降低成员推断风险。

梯度显著性扰动类(CCFA 类)

 优点:对图结构和生成质量影响远小于差分隐私;针对性扰动, 计算开销低;

缺点:仅针对图扩散模型,无法直接迁移到图像扩散模型;对自适 应攻击的鲁棒性待验证。

原生针对图扩散模型,可迁移到图像扩散模型(将边替换为像素/ 特征块)。

七、APDM (Anti-Personalized Diffusion Models) /Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models(APDM)

2025 年,NeurIPS(神经信息处理系统大会,AI 顶会)

扩散模型(生成模型)

防御核心思路:将保护目标从图像转移到模型本身,通过对抗训练

让模型无法对特定主体进行个性化生成,从根源上防止训练数据的隐 私泄露。

模型对抗扰动类(CCFA 类)

优点:防御效果好,不受图像变换影响;可针对特定主体精准保护; 缺点:需要对模型进行对抗微调,增加训练开销;可能影响模型对 非保护主体的生成能力。

原生针对扩散模型设计,完全适配。