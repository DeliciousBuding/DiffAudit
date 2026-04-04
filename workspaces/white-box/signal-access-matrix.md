# 白盒信号访问矩阵

白盒线的关键不是“再找一篇论文”，而是先把允许访问哪些内部信号写清楚。

| 信号 | 需要的访问权限 | 代表论文 | 作用 | 最小资产要求 |
| --- | --- | --- | --- | --- |
| `loss` | 可计算样本级 loss | `White-box MIA` | 最基础的白盒成员信号 | checkpoint、推理脚本、样本输入 |
| `gradient` | 可做每样本梯度或梯度范数计算 | `White-box MIA` | 当前最直接的强白盒信号 | checkpoint、训练配置、梯度接口 |
| `activations` | 可读取中间层激活 | `Finding NeMo` | 定位记忆相关区域 | checkpoint、中间层 hook |
| `cross-attention` | 可读取并干预 cross-attention 神经元 | `Finding NeMo` | 定位具体神经元与记忆机制 | 完整模型、层级命名、干预脚本 |
| `score direction` | 可读取去噪过程中的 score 或方向信息 | `Tracking Memorization Geometry` | 做过程级记忆检测 | 去噪轨迹访问接口 |
| `influence / curvature` | 可算样本影响分数、Hessian 或近似量 | `Layer-wise Influence Tracing` | 更偏审计与缓解，不只是判别 | 训练样本候选集、强内部访问 |

## 当前建议

1. 白盒首批实现只围绕 `gradient` 和 `activations` 两类信号。
2. `Finding NeMo` 更适合作为“记忆定位”扩展，而不是第一批复现入口。
3. 在结果 schema 里提前预留：`sample_id`、`timestep`、`loss`、`grad_norm`、`layer_id`、`neuron_id`。
