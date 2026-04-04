# 黑盒论文矩阵（2024-2026）

这份矩阵用于把黑盒相关论文统一到同一套维度下，避免只剩标题列表。

| 论文 | 年份 | 设定 | 查询对象 | 可见输出 | 额外资产 | 主要指标 | 更适合映射到仓库哪条线 | 当前建议 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2025-324-paper` | 2025 | fine-tuned diffusion 黑盒 MIA | 图像或 prompt 驱动查询 | 最终生成结果 | 微调模型访问接口 | AUC、ASR、成员判别效果 | `black-box` | 当前主复现主线 |
| `Towards Black-Box Membership Inference Attack for Diffusion Models` | 2024 | 纯 API black-box | image variation 或等价输入 | 输出图像 | 无需内部信号 | AUC、TPR/FPR | `black-box` | 做 API-only baseline |
| `CLiD` | 2024 | prompt-conditioned black-box / gray-box | 图文对 | 条件似然差异或等价评分 | prompt、文本条件 | AUC、TPR@FPR | `black-box` / `gray-box` | 做 text-to-image 扩展 |
| `CDI` | 2025 | 数据集级审计 / copyright identification | 一组查询样本 | 聚合成员性证据 | 数据集级候选样本集合 | 数据集级判定 | `gray-box`，次选 `black-box` | 作为审计证据链扩展 |
| `Noise as a Probe` | 2026 | 可控初始噪声下的 MIA | 噪声 + 查询样本 | 最终生成或中间响应 | 初始噪声控制能力 | AUC、低 FPR 指标 | `gray-box` | 先做观察名单，再决定是否接主线 |
| `SIDE` | 2024 | 训练数据提取 / 理论背景 | 无条件采样与提取流程 | 候选重构样本 | 更强攻击假设 | 提取成功率、泄露证据 | `gray-box` / `white-box` 背景 | 用作边界与泄露补充阅读 |

## 统一术语

- `pure API black-box`：只能提交请求，看最终生成结果。
- `prompt-conditioned black-box`：可用 prompt、caption 或条件输入，但不能看模型内部。
- `noise-controllable black-box`：能控制初始噪声或等价 latent 初始化，假设强于普通托管 API。
- `dataset-level audit`：不是单样本成员判别，而是聚合多样本证据去做版权或训练数据集级判断。
- `training-data extraction`：关注是否能直接抽取训练样本，而不是只做成员二分类。

## 当前仓库建议

1. 黑盒主线优先保留 `2025-324-paper`、`Towards Black-Box...`、`CLiD`。
2. `CDI` 用来补“审计证据整理”而不是替代当前 MIA 主线。
3. `Noise as a Probe` 和 `SIDE` 暂时作为扩展阅读，先明确 threat model 再决定是否实现。
