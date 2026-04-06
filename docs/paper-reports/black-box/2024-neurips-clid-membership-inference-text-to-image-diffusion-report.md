# 基于条件似然差异的文生图扩散模型成员推断
Membership Inference on Text-to-image Diffusion Models via Conditional Likelihood Discrepancy

## 文献信息

- 英文标题：Membership Inference on Text-to-image Diffusion Models via Conditional Likelihood Discrepancy
- 中文标题：基于条件似然差异的文生图扩散模型成员推断
- 作者：Shengfang Zhai，Huanran Chen，Yinpeng Dong，Jiajun Li，Qingni Shen，Yansong Gao，Hang Su，Yang Liu
- 发表信息：NeurIPS 2024，arXiv:2405.14800v3
- 论文主问题：在文本到图像扩散模型中，如何利用图文条件关系而非单纯图像误差，稳定判断一条图文样本是否属于训练集
- 威胁模型类别：灰盒查询式成员推断，与黑盒数据使用审计路线相邻，但不是严格的 output-only 黑盒
- 本地 PDF 路径：`D:\Code\DiffAudit\Project\references\materials\black-box\2024-neurips-clid-membership-inference-text-to-image-diffusion.pdf`
- GitHub PDF：[2024-neurips-clid-membership-inference-text-to-image-diffusion.pdf](https://github.com/DeliciousBuding/DiffAudit/blob/main/references/materials/black-box/2024-neurips-clid-membership-inference-text-to-image-diffusion.pdf)
- OCR/Markdown 精修版链接：[OCR精修版：Membership Inference on Text-to-image Diffusion Models via Conditional Likelihood Discrepancy](https://www.feishu.cn/docx/TEmRdXDD2oYJ5Cxn1uFcznIxnZc)
- 飞书原生 PDF：[2024-neurips-clid-membership-inference-text-to-image-diffusion.pdf](https://ncn24qi9j5mt.feishu.cn/file/Df7bbbOCZo3fflxq9PNcI0Fsn3b)
- 开源实现链接：[zhaisf/CLiD](https://github.com/zhaisf/CLiD)
- 报告状态：本地重写稿

## 1. 论文定位

这篇论文研究的是文生图扩散模型的成员推断问题，目标不是恢复训练样本，而是为数据版权所有者提供“某条图文样本是否被用于训练”的审计证据。就 DiffAudit 的文献分层而言，它属于黑盒路线的近邻论文：攻击者不读取模型参数，但需要访问扩散过程中的中间噪声预测，并能对同一图像在不同文本条件下重复查询，因此更准确地说是灰盒查询式攻击。

论文的贡献点不在接口设定本身，而在信号设计。作者指出，对文本到图像模型来说，单看图像侧的误差或似然很难稳定暴露成员性，真正更强的泄露信号来自图文条件关系是否被模型过度记忆。这一点直接把问题从“图像是否被记住”改写为“文本条件是否为该图像提供了异常强的解释力”。

## 2. 核心问题

论文试图回答两个相互关联的技术问题。第一，为什么已有扩散模型成员推断方法迁移到文生图模型后会明显失效，尤其是在较真实的训练步数和默认图像增强下，基线方法常常只能得到接近随机的区分能力。第二，如果文本条件本身携带更强的成员信号，能否在不依赖大量 shadow models 的前提下，把这种信号转化为低查询成本、可校准的成员指标。

作者的回答是肯定的。其核心判断是：文本到图像扩散模型更容易过拟合条件分布 `p(x|c)`，而不是单纯过拟合图像边缘分布 `p(x)`；因此成员推断应当比较“完整文本条件”和“削弱文本条件”之间的似然差异，而不是只比较单一条件下的误差高低。

## 3. 威胁模型与前提

论文采用标准安全游戏定义成员推断。攻击者拿到待审计图文样本 `(x, c)`，并可多次查询目标模型在扩散过程中的中间输出；查询时既可以输入原始文本 `c`，也可以输入空文本 `c_null` 或若干削弱文本 `c_i^*`。攻击者不知道真实的 member/hold-out 划分，但拥有同分布辅助数据，可训练 shadow model 来校准阈值、融合权重和 `CLiDvec` 的分类器。

这一设定的边界也很明确。论文并不适用于只暴露最终生成图像的纯 API 黑盒接口，因为核心指标依赖噪声预测误差。文中还给出一个更弱设定：若拿不到原始文本，可先用 BLIP 生成 pseudo-text 再做推断，但此时效果会明显下降。换言之，论文结论主要适用于具备中间查询能力、且能获得图文配对样本的审计场景。

## 4. 方法总览

作者从“条件过拟合”现象出发构造攻击。直觉上，如果某个图文样本参与过训练，那么正确文本条件通常会显著降低该样本在扩散过程中的噪声预测误差；一旦把文本削弱、截断或置空，这种优势会下降得更明显。相反，非成员样本对文本条件的依赖更接近模型的普通泛化行为，因此完整条件和削弱条件之间的差距不会那么大。

基于这一观察，论文提出条件似然差异 `CLiD`。它不直接估计样本是否“高似然”，而是估计完整条件相对削弱条件为该样本带来的额外解释力。与已有 query-based 方法相比，`CLiD` 的实质变化是把成员信号从图像侧误差转移到图文条件带来的相对增益上，再用 ELBO 差值降低扩散训练噪声带来的随机性。

## 5. 方法概览 / 流程

具体流程可以概括为四步。首先，对原始文本构造削弱条件集合 `C={c_1^*,...,c_k^*}`，其中既包含空文本，也包含若干有控制地删减语义的信息版本。其次，在若干固定扩散时间步上，对完整条件 `c` 与每个削弱条件 `c_i^*` 分别查询目标模型，估计噪声预测误差。然后，计算多个条件似然差 `D_{x,c,c_i^*}` 以及条件似然近似 `L_{x,c}`。最后，将这些量做鲁棒缩放和融合，得到阈值版攻击 `CLiDth`；或者把它们拼接为低维特征向量，再交给 `XGBoost` 得到 `CLiDvec`。

实现上，作者把时间步固定在 `440、450、460`，令 `M=N=3`、`k=4`，因此单样本查询数为 `15`。这一配置的目标不是追求最小查询，而是在随机性、成本和稳定性之间取一个工程上可用的折中。

## 6. 关键技术细节

论文先把“条件过拟合强于边缘过拟合”写成可检验的不等式。设 `q_mem` 与 `q_out` 分别表示成员集和留出集分布，`p` 表示目标模型分布，则作者假设

$$
\mathbb{E}_{\mathbf{c}}\!\left[D\!\left(q_{out}(\mathbf{x}|\mathbf{c}),p(\mathbf{x}|\mathbf{c})\right)-D\!\left(q_{mem}(\mathbf{x}|\mathbf{c}),p(\mathbf{x}|\mathbf{c})\right)\right]
\ge
D\!\left(q_{out}(\mathbf{x}),p(\mathbf{x})\right)-D\!\left(q_{mem}(\mathbf{x}),p(\mathbf{x})\right).
$$

这一步的作用是把经验观察固定为方法前提：成员与非成员之间更大的差异，出现在“文本是否正确匹配图像”这一层，而不是无条件图像分布这一层。论文用 MS-COCO 上的 FID 结果验证了这一点，并显示一旦逐步截断文本，member 与 hold-out 之间的差距会明显缩小。

![条件过拟合关键图](../assets/black-box/2024-neurips-clid-membership-inference-text-to-image-diffusion-key-figure-1-p4.jpeg)

上图对应论文的经验起点。左侧显示在完整文本条件下，member 样本的 FID 明显优于 hold-out；随着文本被截断到 `2/3`、`1/3` 乃至空文本，这种优势持续减弱。右侧进一步显示，member 的 `ΔFID` 下降幅度始终更大，说明模型更依赖它记住的图文绑定关系，而不是只记住图像本身。

在把距离度量取为 KL 散度后，作者得到单样本指标

$$
\mathbb{I}(\mathbf{x},\mathbf{c})=\log p(\mathbf{x}|\mathbf{c})-\log p(\mathbf{x}),
$$

并用条件 ELBO 与空文本 ELBO 的差值来近似它：

$$
\mathbb{I}(\mathbf{x},\mathbf{c})
\approx
\mathbb{E}_{t,\epsilon}\!\left[
\left\|\epsilon_{\theta}(\mathbf{x}_t,t,\mathbf{c}_{null})-\epsilon\right\|_2^2
-
\left\|\epsilon_{\theta}(\mathbf{x}_t,t,\mathbf{c})-\epsilon\right\|_2^2
\right].
$$

这里的关键不是“完整条件误差小”这一件事，而是“完整条件相对削弱条件的误差差值”本身。这样做有两个直接收益：一是把扩散训练本来就很随机的单次 loss 估计改写成相对差值，随机性更低；二是避免分别独立估计两次似然，从而减少查询开销。

在最终攻击器上，`CLiDth` 将多个削弱条件下的差值均值与条件似然 `L_{x,c}` 做鲁棒缩放后融合：

$$
\mathcal{M}_{\text{CLiD}_{th}}(\mathbf{x},\mathbf{c})
=
\mathbf{1}\!\left[
\alpha \cdot \mathcal{S}\!\left(\frac{1}{k}\sum_{i=1}^{k}\mathcal{D}_{\mathbf{x},\mathbf{c},\mathbf{c}_i^*}\right)
+
(1-\alpha)\cdot \mathcal{S}\!\left(\mathcal{L}_{\mathbf{x},\mathbf{c}}\right)
>
\tau
\right].
$$

`CLiDvec` 则把这些差值和 `L_{x,c}` 拼成低维向量，用 `XGBoost` 输出置信分数。实践上，作者比较了简单截断、高斯噪声和 importance clipping 三种条件削弱方式，最终选择 importance clipping，因为它在附录实验里给出了最稳定的 `AUC` 与 `TPR@1%FPR`。

## 7. 实验设置

微调实验覆盖 `Pokemon`、`MS-COCO` 和 `Flickr` 三个数据集，member/hold-out 划分分别为 `416/417`、`2500/2500` 和 `10000/10000`，目标模型为 `Stable Diffusion v1-4`。预训练实验使用 `Stable Diffusion v1-5` 与处理后的 `LAION` 子集。基线包括 `Loss`、`PIA`、`SecMI`、`PFAMI`，以及直接对条件似然做 Monte Carlo 估计的方法。

论文刻意区分两种训练设定。其一是沿用既有工作的 over-training 设定，例如 `MS-COCO` 和 `Flickr` 都训练到 `150000` steps；其二是更接近真实训练脚本的 real-world training 设定，按官方脚本的 step/image ratio 近似为 `20` 重新设置训练步数，并启用默认图像增强 `Random-Crop` 与 `Random-Flip`。评估指标使用 `ASR`、`AUC` 与 `TPR@1%FPR`。所有阈值和分类器都先在 shadow model 上校准，再迁移到目标模型，避免直接窥视目标划分。

## 8. 主要结果

论文首先明确指出，over-training 设定会制造“虚高成功”。在这种设定下，几乎所有方法都能得到很高分数，因而不能说明谁真正更适合真实审计。真正有信息量的是 real-world training 结果：在 `MS-COCO` 上，基线 `AUC` 大多只有 `0.55` 到 `0.65`，而 `CLiDth` 与 `CLiDvec` 分别达到 `0.9613` 与 `0.9630`；在 `Flickr` 上分别达到 `0.9474` 与 `0.9533`；在 `Pokemon` 上也分别达到 `0.9328` 与 `0.9261`。同样重要的是低误报区间，`MS-COCO` 上两种方法的 `TPR@1%FPR` 仍有 `67.52%` 与 `66.36%`，远高于基线的个位数表现。

论文进一步用训练步数轨迹说明，`CLiD` 不是只在严重过拟合时才有效。作者在 `MS-COCO` 的 real-world training 设定下比较不同训练步数，发现 `CLiDth` 在约 `25000` steps 时就已经显著暴露成员信号，而多种基线要到接近 `150000` steps 才达到相近水平。这意味着 `CLiD` 抓住的是更早出现的条件记忆，而不是后期极端过拟合的副产物。

![训练步数轨迹图](../assets/black-box/2024-neurips-clid-membership-inference-text-to-image-diffusion-key-figure-2-p8.jpeg)

这张轨迹图对理解论文主结果很关键。它把“真实训练条件下仍然有效”从表格结论改写成动态过程：成员信号随着训练推进逐步增强，但 `CLiDth` 的上升更早、更陡，说明它检测到的是图文条件绑定的早期记忆。

补充结果同样值得注意。预训练场景下，`CLiDth` 在处理后的 `LAION` 设定上取得 `64.53` 的 `AUC`，仍优于 `PFAMI` 的 `59.08`，但绝对优势显著小于微调场景。拿不到原始文本时，利用 `BLIP` 生成 pseudo-text 后，real-world training 下 `CLiDth` 与 `CLiDvec` 仍有 `83.27` 与 `84.48` 的 `AUC`，说明方法对文本缺失具备一定韧性，但已经明显弱于真实文本条件。

## 9. 优点

这篇论文的技术优点主要有三点。第一，它把文生图模型的成员信号从“图像是否被记住”推进到“图文绑定是否被过度记住”，问题刻画更贴近文本到图像模型的训练目标。第二，作者没有停留在经验直觉上，而是先用条件过拟合现象做分布级验证，再把它推导为单样本指标，方法链路完整。第三，实验对真实训练条件更谨慎，明确揭示了 over-training 会夸大已有方法效果，这一点对审计路线判断尤为重要。

## 10. 局限与有效性威胁

最核心的局限是接口假设偏强。方法要求访问扩散过程中的中间噪声预测，并允许对不同文本条件重复查询，因此不能直接外推到只返回最终生成图像的严格黑盒接口。其次，方法仍依赖同分布辅助数据、shadow model 和文本削弱机制，这些条件在真实平台审计中未必总能满足。再次，预训练实验只有一个处理后的 `LAION` 设定，证据强度弱于微调结论。最后，论文未公开作者实现，某些工程细节只能依靠正文与附录还原。

## 11. 对 DiffAudit 的价值

对 DiffAudit 而言，这篇论文最重要的价值不是直接拿来做黑盒主线，而是提供了一个更强的信号设计原理：如果模型能访问条件文本，那么“条件扰动前后响应差异”往往比单纯图像误差更接近真正的成员信号。它适合进入黑盒路线的邻近支撑材料，用来论证为什么未来的审计接口设计应优先考虑条件对照实验。

在工程层面，这篇论文提示我们：若后续有能力拿到中间 denoiser 响应，最值得优先实现的不是更多图像侧分数函数，而是条件裁剪、差值估计与多时间步聚合。在叙事层面，它也有价值，因为它把成员推断与未授权数据使用审计直接联系起来，较容易衔接版权审计场景。

## 12. 关键图使用方式

- 图 1 用于解释“条件过拟合”这一方法前提。它说明为什么完整文本与削弱文本之间的差异能成为成员信号。
- 图 2 用于解释主结果不只是静态表格领先，而是在训练早期就更快暴露成员性，这直接影响对真实训练场景的判断。

## 13. 复现评估

若要做忠实复现，至少需要 `Stable Diffusion` 微调与查询管线、图文配对数据集、可访问的中间噪声预测接口、shadow model 训练流程、importance clipping 的词重要性计算、以及文本缺失时的 `BLIP` 伪文本生成模块。当前 DiffAudit 仓库真正缺少的是灰盒查询接口，而不是对论文思路的理解；只要接口仍停留在严格 black-box，这篇论文就无法被原样复用。另一个结构性阻塞是作者未公开官方实现，预处理细节与某些默认参数需要从正文和附录自行补齐。

## 14. 写回总索引用摘要

这篇论文解决的是文生图扩散模型中的成员推断问题，目标是在给定图文样本 `(x, c)` 的情况下判断其是否参与过模型训练，并将这一能力用于未授权数据使用审计。

论文的核心方法是利用“条件过拟合”现象，比较完整文本条件与削弱文本条件之间的条件似然差异，构造 `CLiDth` 和 `CLiDvec` 两类攻击器；在更接近真实训练步数且启用默认图像增强的设定下，方法仍显著优于图像侧误差基线。

它对 DiffAudit 的价值在于提供了一条清晰的信号设计方向：若未来审计能力允许访问条件响应或中间查询结果，应优先考虑条件扰动前后的响应差异；但由于接口假设属于灰盒，这篇论文更适合作为黑盒路线的近邻参考，而不是直接主方法。
