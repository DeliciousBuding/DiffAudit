# 白盒方向计划

## 推荐论文

- `qy_White-box Membership Inference Attacks againstDiffusion Models.pdf`
- `SecMI_ICML2023.pdf`

## 起步建议

1. 先钉死白盒定义：
   - 可访问参数
   - 可访问梯度
   - 可访问 loss 或中间状态
2. 明确白盒最小资产清单：
   - 完整 checkpoint
   - 训练期配置
   - 中间信号访问方式
3. 先写实验假设和输入输出，再考虑执行代码

## 当前难点

- 白盒和灰盒边界容易混
- 对 checkpoint 和训练配置要求高
- 复现出来的强结果未必贴近真实部署场景
