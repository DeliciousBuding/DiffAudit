# 统一结果 Schema

这份文档用于约束黑盒、白盒、灰盒三条线的最小共享结果结构。

## 最小字段

| 字段 | 说明 |
| --- | --- |
| `status` | `blocked / ready / error` |
| `track` | `black-box / white-box / gray-box` |
| `method` | 攻击方法，例如 `secmi` |
| `paper` | 对应主论文或实现线索 |
| `mode` | `dry-run / smoke / full-run` |
| `device` | `cpu / cuda` |
| `workspace` | 实验输出目录 |
| `assets` | 数据集、模型与成员划分等资产信息 |
| `checks` | 资产或运行前检查结果 |
| `metrics` | `auc / asr / tpr_at_low_fpr` 等指标 |
| `cost` | 查询次数、timestep、优化步数、GPU-hours |
| `notes` | 对 threat model、限制和异常的说明 |

## 模板位置

- `experiments/templates/attack-result-template.json`

## 当前建议

1. 黑盒和灰盒优先先对齐这套结构。
2. 白盒后续可以在 `assets` 和 `checks` 中增加 `gradient / activation / layer / neuron` 相关字段。
3. 没有真实运行证据时，不要把 `blocked` 或 `dry-run` 输出写成 benchmark 结果。
