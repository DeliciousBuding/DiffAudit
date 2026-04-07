# MIA Defense Document

结构化提取版本。来源于同目录下的 [mia-defense-document.docx](mia-defense-document.docx)，目标是把内部 Word 文档整理成可检索、可引用、可继续加工的 Markdown。

## 文档元信息

- 标题：`研究方向技术文档：针对扩散模型的成员推断攻击防御`
- 副标题：`黑盒 · 白盒 · 灰盒 三维威胁模型下的统一防御框架`
- 版本：`v1.0`
- 日期：`2026年4月`
- 用途：`供研究组内部使用`

## 一、研究背景与问题定义

文档把问题定义为：

- 扩散模型已经广泛部署，训练数据隐私风险会随着商业落地放大。
- 成员推断攻击（Membership Inference Attack, MIA）的核心任务，是判断某个样本是否参与过目标模型训练。
- 医疗影像、人脸数据、版权图像是重点敏感场景。

文档聚焦的核心问题有三类：

1. 扩散模型在黑盒、灰盒、白盒三种攻击设定下，各自的成员泄露机制是什么。
2. 针对每种威胁模型，如何设计有效且实用的防御机制。
3. 去噪轨迹、潜变量、条件信号等扩散模型特有结构，是否引入新的攻击面，以及如何针对性防御。

## 二、三种威胁模型的严格定义

| 属性 | 黑盒（Black-box） | 灰盒（Gray-box） | 白盒（White-box） |
| --- | --- | --- | --- |
| 模型访问 | 仅输出（生成图像、损失值） | 架构已知，部分权重/API | 完整参数、梯度、激活值 |
| 先验知识 | 无模型内部信息 | 知道模型架构和超参数 | 完整模型权重 |
| 主要攻击方式 | 重建误差对比；多步骤损失统计；似然阈值判断 | 影子模型攻击；迁移攻击；元分类器攻击 | 梯度范数分析；DDIM Inversion；中间激活差异 |
| 典型现实场景 | 攻击商业 API（Midjourney 等） | 攻击开源部署模型 | 内部人员 / 模型泄露场景 |

## 三、扩散模型特有攻击面分析

文档将“扩散模型特有攻击面”列为第一个核心创新点，分三类。

### 3.1 去噪轨迹泄露

- 泄露机制：
  - 模型去噪过程包含多步损失序列 `L_1 ... L_T`。
  - 成员样本通常表现为整体更低、更加平滑的损失轨迹。
  - 非成员样本通常表现为更高且波动更大的损失轨迹。
- 文档给出的代表攻击：
  - `SecMI`
- 文档给出的防御思路：
  - 轨迹截断（Trajectory Truncation）
  - 步骤级噪声注入
  - 随机步骤采样

### 3.2 Classifier-Free Guidance 条件信号泄露

- 泄露机制：
  - CFG 同时计算条件输出 `eps_c` 和无条件输出 `eps_u`。
  - 文档认为成员样本的 `eps_c - eps_u` 差值范数通常更大，可作为成员信号。
- 文档给出的防御思路：
  - Guidance Scale 扰动
  - 无条件输出混淆
  - 限制 guidance 信息暴露

### 3.3 DDIM Inversion 潜变量泄露

- 泄露机制：
  - 通过 DDIM Inversion 把图像映射回潜空间噪声编码 `x_T`。
  - 文档认为成员样本的潜编码更贴近训练时分布，因而重建误差更低。
- 文档给出的防御思路：
  - 潜空间随机化
  - 禁止确定性 Inversion API
  - 潜编码差分隐私扰动

## 四、三维威胁模型下的防御方案

文档把“按威胁模型拆分专用防御机制”列为第二个核心创新点。

### 4.1 黑盒防御

目标：让成员 / 非成员的可观测输出分布难以区分。

#### 方法 B-1：输出概率平滑（Output Probability Smoothing）

- 原理：
  - 对模型公开的损失 / 概率值加噪，降低成员判别统计性。
- 实现描述：
  - `L_pub = L + Lap(0, b)`
  - 其中 `b` 由 DP 灵敏度和目标 `epsilon` 决定。
- 文档给出的优点：
  - 与训练过程解耦，可即插即用。
- 文档给出的代价：
  - 噪声过大时会损伤输出可信度。

#### 方法 B-2：查询次数限制与速率控制

- 原理：
  - 黑盒 MIA 往往依赖重复查询降低估计方差。
- 实现描述：
  - 基于感知哈希 `pHash` 追踪相似查询。
  - 超过阈值后拒绝服务或返回降质输出。

### 4.2 白盒防御

目标：在训练阶段抑制成员信号的形成。

#### 方法 W-1：扩散模型专用 DP-SGD（Diffusion-DP）

- 原理：
  - 通过梯度裁剪和噪声注入，限制单个训练样本对参数的影响。
- 文档强调的关键挑战：
  - 扩散模型上 per-sample gradient 代价高。
  - 隐私预算 `epsilon` 与图像质量 `FID` 有明显权衡。
- 文档提出的实现方向：
  - 针对不同时间步使用自适应梯度裁剪阈值。
- 文档写下的目标：
  - `epsilon <= 8`
  - `FID` 降幅控制在 `15%` 以内

#### 方法 W-2：成员信号对抗训练（Adversarial Membership Signal Suppression）

- 原理：
  - 通过附加正则项，逼近成员 / 非成员损失分布。
- 文档给出的训练目标：
  - `L_total = L_denoise + lambda * L_member_align`
  - `L_member_align = max(0, L_member_mean - L_nonmember_mean + margin)`

### 4.3 灰盒防御

目标：降低成员信号在模型间的可迁移性。

#### 方法 G-1：推理时架构随机化（Inference-time Architecture Randomization）

- 原理：
  - 在推理阶段随机化局部结构，使影子模型训练出的元分类器难以对齐目标分布。
- 文档给出的实现方式：
  - 随机 dropout
  - 随机 UNet skip connection 选择
  - 随机种子配置轮换

#### 方法 G-2：知识蒸馏代理模型（Knowledge Distillation Proxy）

- 原理：
  - 对外不直接部署原始训练模型，而是部署在公开数据上蒸馏得到的学生模型。
- 文档给出的目标：
  - 保留生成能力
  - 削弱细粒度成员记忆
  - 同时缓解黑盒与灰盒攻击

## 五、统一评估框架

文档把“跨三种威胁模型的统一评估基准”列为第三个核心创新点。

### 5.1 评估指标

| 指标类别 | 具体指标 | 说明 |
| --- | --- | --- |
| 防御效果 | `ASR`、`AUC-ROC`、`TPR@FPR=0.1%` | 越低越好；`TPR@FPR=0.1%` 被文档视为严格隐私标准 |
| 生成质量 | `FID`、`IS`、`LPIPS` | 防御不应显著降低生成质量 |
| 计算开销 | 训练时间开销、推理延迟、GPU 显存增量 | 评估实际可部署性 |
| 自适应攻击鲁棒性 | 攻击者知晓防御后的 `ASR`、最强自适应攻击 `ASR` | 防御必须在机制已知时仍有效 |

### 5.2 基准数据集与模型

- 数据集：
  - `CIFAR-10`
  - `CelebA-HQ`
  - `ImageNet-64`
- 基准模型：
  - `DDPM`
  - `DDIM`
  - `Stable Diffusion v1.5`
- 攻击基准：
  - `Carlini 2021`
  - `SecMI`
  - `GSA`

## 六、任务分工建议

| 负责人 | 威胁模型 | 核心任务 | 预期产出 |
| --- | --- | --- | --- |
| `A` | 黑盒 | 复现 `SecMI` 和 `Carlini 2021`；实现 `B-1` 与 `B-2`；评估去噪轨迹泄露场景下的防御效果 | 黑盒攻击/防御实验结果；轨迹截断消融；隐私-效用权衡曲线 |
| `B` | 白盒 | 实现 `W-1`、步骤自适应裁剪、`W-2`；分析 `DDIM Inversion` 泄露与防御 | `Diffusion-DP` 训练代码；裁剪消融；白盒防御评估 |
| `C` | 灰盒 | 搭建影子模型攻击基准；实现 `G-1` 与 `G-2`；评估 CFG 条件信号泄露防御 | 灰盒攻击/防御结果；蒸馏代理模型质量分析；跨数据集迁移验证 |
| `全组协作` | 三盒统一 | 统一评估框架；自适应攻击验证；论文实验部分撰写 | 开源代码库；完整实验表；论文初稿 |

## 七、关键参考文献

### MIA 基础文献

1. Carlini et al. Membership Inference Attacks From First Principles. IEEE S&P 2022.
   - 本地文件：[2022-ieee-membership-inference-first-principles.pdf](../survey/2022-ieee-membership-inference-first-principles.pdf)
2. Shokri et al. Membership Inference Attacks Against Machine Learning Models. IEEE S&P 2017.
   - 本地文件：[2017-ieee-membership-inference-machine-learning-models.pdf](../survey/2017-ieee-membership-inference-machine-learning-models.pdf)

### 扩散模型 MIA 攻击

3. Duan et al. Are Diffusion Models Vulnerable to Membership Inference Attacks? ICML 2023.
   - 本地文件：[2023-icml-secmi-membership-inference-diffusion-models.pdf](../gray-box/2023-icml-secmi-membership-inference-diffusion-models.pdf)
4. Wu et al. Membership Inference of Diffusion Models. SecMI. 文档写为 NeurIPS 2023。
   - 本地文件：[2023-icml-secmi-membership-inference-diffusion-models.pdf](../gray-box/2023-icml-secmi-membership-inference-diffusion-models.pdf)
5. Matsumoto et al. Membership Inference Attacks Against Diffusion Models. IEEE S&P Workshop 2023.
   - 本地文件：[2023-ieee-spw-membership-inference-attacks-diffusion-models.pdf](../survey/2023-ieee-spw-membership-inference-attacks-diffusion-models.pdf)

### 差分隐私与扩散模型

6. Dockhorn et al. Differentially Private Diffusion Models. TMLR 2023.
   - 本地文件：[2023-tmlr-differentially-private-diffusion-models.pdf](../survey/2023-tmlr-differentially-private-diffusion-models.pdf)
7. Ghalebikesabi et al. Differentially Private Diffusion Models Generate Useful Synthetic Images. arXiv 2023.
   - 本地文件：[2023-arxiv-differentially-private-diffusion-models-generate-useful-synthetic-images.pdf](../survey/2023-arxiv-differentially-private-diffusion-models-generate-useful-synthetic-images.pdf)

### 通用 MIA 防御

8. Jia et al. MemGuard: Defending Against Black-Box Membership Inference Attacks via Adversarial Examples. CCS 2019.
   - 本地文件：[2019-ccs-memguard-defending-black-box-membership-inference.pdf](../survey/2019-ccs-memguard-defending-black-box-membership-inference.pdf)
9. Nasr et al. Machine Learning with Membership Privacy using Adversarial Regularization. CCS 2018.
   - 本地文件：[2018-ccs-membership-privacy-adversarial-regularization.pdf](../survey/2018-ccs-membership-privacy-adversarial-regularization.pdf)

## 附录：三大创新点概要

### 贡献一：扩散模型新型攻击面的系统刻画

文档声称首次系统分析三类成员泄露渠道：

- 多步去噪轨迹
- Classifier-Free Guidance 条件信号
- DDIM Inversion 潜变量

### 贡献二：三维威胁模型下的专用防御机制

文档提出的三类防御族包括：

- 黑盒：`B-1` 输出概率平滑，`B-2` 查询限速
- 白盒：`W-1` Diffusion-DP，`W-2` 成员信号对抗训练
- 灰盒：`G-1` 推理时架构随机化，`G-2` 知识蒸馏代理模型

### 贡献三：统一评估基准与开源代码库

文档的目标终点是：

- 建立覆盖黑盒、灰盒、白盒的统一评估框架
- 覆盖多数据集、多攻击、多防御
- 形成开源代码库与论文初稿
