# 白盒方向计划

## 推荐论文

- `WhiteBox_PoPETs2025.pdf`
- `qy_White-box Membership Inference Attacks againstDiffusion Models.pdf`
- `SecMI_ICML2023.pdf`

## 推荐定位

- 主论文：`WhiteBox_PoPETs2025.pdf`
- 基础参照：`SecMI_ICML2023.pdf`
- 对照阅读：`2025-324-paper.pdf`

## 起步建议

1. 先明确白盒边界：哪些内部信号可以访问
2. 先列资产清单：checkpoint、训练配置、loss、gradient、中间状态
3. 先写实验假设和指标，再决定执行顺序
4. 先从最小可验证路径开始，不要一上来追完整论文指标

## 当前难点

- 白盒和灰盒边界容易混
- 对 checkpoint 和训练配置要求高
- 复现结果虽然通常更强，但未必贴近真实部署场景
