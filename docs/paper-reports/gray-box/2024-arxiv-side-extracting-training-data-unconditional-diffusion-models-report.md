# SIDE：利用代理条件从扩散模型提取训练数据

SIDE: Surrogate Conditional Data Extraction from Diffusion Models

## 文献信息

- 英文标题：SIDE: Surrogate Conditional Data Extraction from Diffusion Models
- 中文标题：SIDE：利用代理条件从扩散模型提取训练数据
- 作者：Yunhao Chen，Shujie Wang，Difan Zou，Xingjun Ma
- 发表 venue / year / version：arXiv 2024；manifest 记录的材料版本对应 arXiv:2410.02467v7，日期为 2025-08-01
- 论文主问题：在没有显式 prompt 或类别标签时，是否仍能从无条件扩散模型中定向提取训练样本
- 威胁模型类别：主体是白盒训练数据提取；附录补充了黑盒查询式扩展与投毒微调后的后门提取扩展
- 材料索引路径：`references/materials/gray-box/2024-arxiv-side-extracting-training-data-unconditional-diffusion-models.pdf`
- 上游来源 URL：见 `references/materials/manifest.csv` 中对应的 `source_url` 字段
- OCR 精修版链接：[OCR精修版：SIDE: Surrogate Conditional Data Extraction from Diffusion Models](https://www.feishu.cn/docx/RfqVdBxQfokTDzxjKsxcj9wWnJe)
- 飞书原生 PDF：[2024-arxiv-side-extracting-training-data-unconditional-diffusion-models.pdf](https://ncn24qi9j5mt.feishu.cn/file/C1L2bJ2T9oOr6Kx1fQ7cvzlBnYx)
- 开源实现：暂未找到官方代码
- 报告状态：已完成

## 1. 论文定位

这篇论文属于训练数据提取攻击，而不是成员推断。它针对扩散模型记忆与可提取性的关系提出一个更强的命题：无条件扩散模型之所以看起来更安全，不是因为它们不会记忆，而是因为现有攻击缺少足够精确的条件信号，无法把采样过程稳定推向记忆样本所在的高密度区域。

对 DiffAudit 而言，这篇论文不应被当成当前灰盒主线的直接基线，因为主体设定要求白盒参数访问；更合适的定位是 gray-box 路线的桥接文献和边界材料。它把 side information 的来源从外部 prompt、类别标签推进到模型自生成样本里可恢复的聚类结构。

## 2. 核心问题

论文试图回答两个连续的问题。第一，为什么条件扩散模型通常比无条件模型更容易暴露训练样本。第二，如果无条件模型内部同样形成了稳定聚类，那么攻击者能否先恢复这些簇，再把簇中心当作 surrogate condition，把无条件采样改写为可定向的训练数据提取过程。

作者给出的回答是肯定的。关键不在于“是否存在天然标签”，而在于“能否构造足够 informative 的条件”。只要条件足够窄，模型就会更容易落入某个局部高密度区域，进而放大记忆与提取成功率。

## 3. 威胁模型与前提

主体攻击是白盒设定。攻击者能够访问目标 DPM 参数、采样器和中间 score 相关量，但并不持有原始训练集标签。攻击者额外需要一个预训练特征提取器、一批由目标模型自己生成的合成图像，以及足够的算力来训练时间相关分类器或 LoRA 适配器。

论文的主结论适用于“参数可见、可重采样、可额外训练代理模块”的边界；因此它不等价于商业 API 场景。附录虽给出基于遗传算法的黑盒 proof-of-concept，但其查询成本极高，只能说明可行性，不能证明实际可部署性。

## 4. 方法总览

SIDE 的直觉是先制造条件，再利用条件。攻击者首先从目标 DPM 采样一批合成图像，用预训练特征提取器把它们嵌入语义空间，再执行 K-means 聚类，并剔除 cohesion 过低的簇。剩余簇的中心被当作 surrogate condition，它们不是人工标签，而是目标模型内部数据结构的外显代理。

得到 surrogate condition 后，攻击者再把它接入反向去噪过程。对小模型，论文训练时间相关分类器，并在每个反向步加入条件梯度；对大模型，如 Stable Diffusion 1.5，则冻结原模型主体，仅通过 LoRA 在伪标签合成数据上做条件化微调。最终攻击时随机挑一个目标簇，按该簇对应的 surrogate label 反复采样，从而向高密度、易记忆的局部区域逼近。

![SIDE 方法直觉图](../assets/gray-box/2024-arxiv-side-extracting-training-data-unconditional-diffusion-models-key-figure-1-p2.png)

这张图最关键的贡献是把三种条件方式放到同一画面里比较。无条件采样只能在整个数据区域内漫游；普通条件模型虽然有 prompt 或类别标签，但语义区域仍然过宽；SIDE 通过聚类得到更窄的 surrogate condition，把搜索空间压缩到更接近记忆样本的局部密集区域。

## 5. 方法概览 / 流程

方法流程可以概括为两阶段。第一阶段是“构造代理条件”：采样合成图像，提取特征，聚类，过滤低 cohesion 簇，并给合成图像分配伪标签。第二阶段是“把代理条件接回去噪过程”：若目标是小模型，就训练时间相关分类器；若目标是大模型，就用 LoRA 学一个带伪标签条件的轻量条件模型。最后按目标簇做引导采样，再与训练集计算相似度，得到提取结果与 AMS/UMS 等指标。

与 Carlini 等基线相比，SIDE 的真正变化不是换了一个更强的相似度度量，而是把攻击入口从“直接对模型做无条件或粗条件采样”改成“先恢复内部簇结构，再把簇结构当成条件”。它把条件接口从外部提供的 prompt，换成了模型自己暴露出的内部分组。

## 6. 关键技术细节

第一条关键技术线是 surrogate guidance。结合正文描述与 Algorithm 1，可以合理判断论文实际使用了 guidance scale `\lambda` 来调节条件梯度强度；born-digital Markdown 抽取出的公式漏掉了这个系数，但算法输入里显式包含 `guidance scale λ`。对应的采样形式可写为

$$
dx=\left[f(x,t)-g(t)^2\left(\nabla_x \log p_t^\theta(x)+\lambda \nabla_x \log p_t^\theta(y_I \mid x)\right)\right]dt+g(t)dw.
$$

这里的关键不只是 classifier guidance 本身，而是条件 `y_I` 并不来自人工标签，而是来自目标模型自生成样本的聚类中心。论文的核心判断是：如果这些簇能更紧地隔离某一组相似样本，模型的记忆就会被更强地放大。

第二条关键技术线是分布级记忆度量。论文没有只用单张图最近邻命中来定义 memorization，而是引入一个近似训练集经验分布与生成分布之间的 KL 散度：

$$
\mathcal{M}(\mathcal{D};p_\theta,\epsilon)=D_{\mathrm{KL}}(q_\epsilon \Vert p_\theta), \qquad
q_\epsilon(x)=\frac{1}{N}\sum_{x_i\in\mathcal{D}}\mathcal{N}(x\mid x_i,\epsilon^2 I).
$$

这个量越小，说明生成分布越贴近训练样本支撑集。它的作用不是直接做攻击评分，而是给后面的 informative label 理论提供一个模型级解释框架，即“条件是否能把模型推向更窄的子分布”。

第三条关键技术线是评测指标。论文提出 AMS 与 UMS，分别衡量“生成样本中有多少命中了训练样本的相似度区间”和“命中了多少个不同训练样本”：

$$
\mathrm{AMS}(\mathcal{D}_1,\mathcal{D}_2,\alpha,\beta)=
\frac{\sum_{x_i\in\mathcal{D}_1}\mathcal{F}(x_i,\mathcal{D}_2,\alpha,\beta)}{N_G},
\qquad
\mathrm{UMS}(\mathcal{D}_1,\mathcal{D}_2,\alpha,\beta)=
\frac{\left|\bigcup_{x_i\in\mathcal{D}_1}\phi(x_i,\mathcal{D}_2,\alpha,\beta)\right|}{N_G}.
$$

AMS 更偏向 hit rate，UMS 更偏向 unique extraction。论文用 low、mid、high similarity 三档来区分近似复制、较强语义相似和更高保真度命中，这比单纯报告一个 `95th percentile SSCD` 更能解释攻击到底在“撞到多少”与“撞到多少不同样本”两个维度上表现如何。

## 7. 实验设置

- 数据集：CIFAR-10，CelebA-HQ-FI，CelebA-25000，CelebA，ImageNet，LAION-5B。
- 模型：前五个数据集使用从头训练的扩散模型；LAION-5B 使用预训练 Stable Diffusion 1.5。
- 基线：Carlini UnCond 与 Carlini Cond。
- 指标：低 / 中 / 高相似度下的 AMS、UMS，以及高分辨率数据上的 `95th percentile SSCD` 和低分辨率数据上的 `95th percentile L2 distance`。
- 关键条件：100 个聚类，cohesion threshold 为 `0.5`，ResNet34 伪标签器，Stable Diffusion 的 LoRA rank 为 `512`，分类器学习率 `1e-4`，LoRA 学习率 `1e-5`。

## 8. 主要结果

论文的主结论很直接：SIDE 在六个数据集上全部超过 Carlini UnCond，并且系统性超过 Carlini Cond。这说明“天然存在 prompt 或类别标签”并不是最关键的，真正关键的是条件是否足够精确。

最能说明问题的数字包括：CelebA-25000 上，SIDE 的 low-similarity AMS 为 `20.527%`，显著高于 Carlini Cond 的 `8.712%`；CelebA-HQ-FI 上，SIDE 的 mid-similarity AMS / UMS 为 `2.227% / 0.842%`，高于 Carlini Cond 的 `1.310% / 0.554%`；LAION-5B 上，SIDE 的 `95th percentile SSCD` 为 `0.394`，也明显高于条件基线的 `0.253`。即使在最难的 ImageNet 上，SIDE 也把 low-similarity AMS 从 `0.152%` 提升到 `0.443%`。

![SIDE 抽取样例图](../assets/gray-box/2024-arxiv-side-extracting-training-data-unconditional-diffusion-models-key-figure-2-p5.png)

这张图不是最强的定量证据，但它补充了表 1 无法直接展示的现象：SIDE 不只是生成“看起来像某一类”的样本，而是能在 low、mid、high 三个相似度区间逐步逼近训练图像。换言之，作者不是把广义语义相似误写成复制，而是明确按相似度分档展示了抽取强度。

## 9. 优点

这篇论文的技术优点主要有三点。第一，它把“条件会放大记忆”从经验现象提升成 informative label 框架，而不是只停留在 prompt / 类别标签层面。第二，它同时给出了小模型和大模型两条实现路径，分别对应时间相关分类器和 LoRA 条件化。第三，评测不只报告单个分位数，而是用 AMS / UMS 把 hit rate 与 unique extraction 分开，实验解释力更强。

## 10. 局限与有效性威胁

第一，主体结论建立在白盒访问之上，离 DiffAudit 当前更关心的 gray-box / API 审计还有距离。第二，黑盒附录虽然证明了可行性，但代价很高：跑到 `800` 代需要 `40,000` 次查询，high-similarity UMS 也只有 `0.010%`，因此不能把它等同于可实用的黑盒攻击。第三，AMS / UMS 是论文自定义指标，解释力强，但横向对比其他工作时需要注意口径差异。第四，论文还加入了投毒微调后的后门提取扩展，这条路线与主体攻击的前提不同，不能混成同一个威胁模型来讨论。

## 11. 对 DiffAudit 的价值

这篇论文对 DiffAudit 的价值，不在于它已经提供了一个可直接复用的灰盒基线，而在于它重新定义了 side information。此前不少工作把条件信号理解为外部 prompt、类别标签或身份 cue；SIDE 说明，模型自己生成样本中的聚类结构也可以被逆向恢复并重新用作条件。

因此它更适合作为 gray-box 路线的桥接文献和叙事支点。它提醒我们，不应把“没有显式条件接口”误判为“泄露风险低”；同时，它也为后续 beyond-membership 的训练样本提取、证据分层和指标设计提供了理论背景。若要进入工程实现，它更像一篇需要拆解和降假设的上游参考，而不是今天就能复现的最低成本实验。

## 12. 关键图使用方式

本报告只保留两张图。第一张放在方法总览后，用于解释 surrogate condition 相比无条件采样和普通条件采样为何更容易逼近记忆样本。第二张放在主结果后，用于展示 low、mid、high similarity 三档样例，帮助读者把表 1 的分档指标与视觉现象对应起来。这里没有再单独插入超参数曲线或附录黑盒表，因为展示稿重点是讲清主机制和主证据。

## 13. 复现评估

忠实复现 SIDE 需要目标 DPM 权重、可控采样器、预训练特征提取器、聚类和 cohesion 过滤模块、时间相关分类器或 LoRA 微调代码、以及能和训练集逐样本比较的评测管线。论文提供了 cluster count、cohesion threshold、LoRA rank、学习率和生成样本数，因此方向性复现并不神秘。

真正的结构性阻塞有三个。第一，未见官方代码。第二，AMS / UMS 依赖对训练集的逐样本比对，数据与相似度实现必须齐备。第三，黑盒附录与后门扩展都需要额外工程组件。结合当前仓库，更现实的做法是先把 SIDE 作为 threat modeling 与指标设计参考，而不是立即承诺全文数值复现。

## 14. 写回总索引用摘要

这篇论文解决的是扩散模型训练数据提取中的一个关键误区：无条件扩散模型之所以看似更安全，并不意味着它们不会记忆，而是现有攻击缺少足够精确的条件信号。

作者提出 SIDE，先从目标模型自生成样本中恢复聚类结构，再把簇中心当作 surrogate condition，通过时间相关分类器或 LoRA 条件化把采样过程推向记忆样本所在的高密度区域。实验表明，SIDE 在 CIFAR-10、CelebA、ImageNet 和 LAION-5B 上都超过既有无条件与条件抽取基线。

它对 DiffAudit 的价值主要在于扩展了 side information 的定义，把灰盒相关路线从“是否拿到 prompt / label”推进到“是否能恢复模型内部聚类结构”这一层，因此是灰盒桥接叙事和指标设计的重要背景文献。
