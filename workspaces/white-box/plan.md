# 白盒方向计划

## 状态面板

- `owner`: 待认领
- `scope`: 白盒成员推断、记忆定位、梯度与激活级审计
- `status`: 调研中
- `blocked by`: 可访问的 checkpoint、训练配置、每样本梯度或中间激活接口
- `next command`: `Get-Content workspaces/white-box/plan.md`
- `last updated`: 2026-04-05

## 推荐论文

- `2025-popets-white-box-membership-inference-diffusion-models.pdf`
- `2025-local-mirror-white-box-membership-inference-diffusion-models.pdf`
- `2023-icml-secmi-membership-inference-diffusion-models.pdf`
- `2024-neurips-finding-nemo-localizing-memorization-neurons-diffusion-models.pdf`

## 推荐定位

- 主论文：`2025-popets-white-box-membership-inference-diffusion-models.pdf`
- 基础参照：`2023-icml-secmi-membership-inference-diffusion-models.pdf`
- 对照阅读：`2025-ndss-black-box-membership-inference-fine-tuned-diffusion-models.pdf`
- 记忆定位扩展：`Finding NeMo`

## 起步建议

1. 先明确白盒边界：哪些内部信号可以访问
2. 先列资产清单：checkpoint、训练配置、loss、gradient、中间状态
3. 补白盒信号访问矩阵：`loss / gradient / activations / cross-attention / score direction`
4. 先写实验假设和指标，再决定执行顺序
5. 先从最小可验证路径开始，不要一上来追完整论文指标

## 当前难点

- 白盒和灰盒边界容易混
- 对 checkpoint 和训练配置要求高
- 复现结果虽然通常更强，但未必贴近真实部署场景
