# 白盒方向初始计划

## 方向定位

白盒方向关注：

- 可访问模型内部参数或中间信号
- 可利用 loss、gradient、intermediate states 等内部信息
- 目标通常是获得更强攻击信号

## 当前建议主论文

- 主论文：`qy_White-box Membership Inference Attacks againstDiffusion Models.pdf`

## 第一周目标

1. 明确白盒设定里允许访问哪些内部信号
2. 梳理论文使用的攻击特征
3. 列出所需资产和环境要求
4. 写出白盒最小实验假设

## 当前阻塞

- 尚未落下统一白盒 adapter
- 需要明确 checkpoint、梯度与中间状态访问方式
