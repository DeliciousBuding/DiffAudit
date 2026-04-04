# 灰盒 Observable Contract

灰盒线必须先定义“能看见什么”，否则论文之间无法公平比较。

## 权限分层

| 等级 | 可见信息 | 代表论文 | 备注 |
| --- | --- | --- | --- |
| `L1` | 只有最终 loss 或等价评分 | `PIA` 的弱权限理解 | 最接近最弱灰盒 |
| `L2` | `epsilon_t`、噪声预测或特定 timestep 响应 | `PIA` | 典型部分中间信息 |
| `L3` | 部分去噪 latent 或中间样本 | `Structural Memorization` | 已明显强于纯黑盒 |
| `L4` | 条件似然差异、caption 条件接口 | `CLiD` | 适合 text-to-image |
| `L5` | caption 缺失但可优化 surrogate embedding | `No Caption, No Problem` | 更接近现实缺 caption 场景 |

## caption 维度

- `caption-known`：攻击者知道 prompt 或 caption。
- `caption-unknown`：攻击者不知道原始文本，需要 surrogate 或 fitted embedding。

这两种情况不要混着比较，应当分开记录。

## 成本 ledger 建议字段

- `queries_per_sample`
- `timesteps_observed`
- `optimization_steps`
- `extra_caption_or_vlm_cost`
- `gpu_hours`
- `auc`
- `tpr_at_low_fpr`

## 当前建议

1. 灰盒首批主线保留 `SecMI -> PIA -> CLiD`。
2. `Structural Memorization` 用来补 `L3` 层的中间结构信号。
3. `No Caption, No Problem` 用来补 `caption-unknown` 轨道，不要和 `CLiD` 直接混在一起比较。
