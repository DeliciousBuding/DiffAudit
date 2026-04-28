# 仓库地图

这份文档用于说明当前仓库各目录的职责，减少“该把内容放在哪里”的摩擦。

## 顶层目录

| 路径 | 职责 |
| --- | --- |
| `src/diffaudit/` | 共享可执行代码，所有 CLI、planner、probe、dry-run、smoke 逻辑都应该落在这里 |
| `tests/` | 自动化测试；新增命令、planner、adapter 和 smoke 默认都要先补测试 |
| `configs/` | 可版本化配置模板；不要把个人机器路径直接写进共享配置 |
| `experiments/` | 可提交的最小运行证据，如 `summary.json`；不要提交大体积中间产物 |
| `references/` | 论文 PDF、索引、评分和来源元数据 |
| `docs/` | 项目级说明文档、状态总览、目录导航和协作规则 |
| `workspaces/` | 多人协作工作区；方向计划、背景笔记、阻塞记录和阶段总结 |
| `external/` | 本地浅克隆或探索代码，上游仓库上下文，不纳入版本控制 |
| `third_party/` | 已被裁剪并纳入版本控制的上游依赖子集 |
| `<DIFFAUDIT_ROOT>/Download/` | 仓库外原始资产层；放数据集、权重、supplementary 包和大文件 manifest |

## `src/diffaudit/`

| 路径 | 职责 |
| --- | --- |
| `cli.py` | 所有仓库对外命令入口 |
| `config.py` | 统一配置 dataclass 与 YAML 加载 |
| `attacks/` | 各方法线的 planner、asset probe、dry-run、runtime smoke 和 artifact summary |
| `pipelines/` | 跨方法通用 pipeline，目前主要是最小 smoke pipeline |
| `metrics/` | 预留给统一指标计算逻辑 |
| `reports/` | 预留给统一报告输出层 |
| `assets/` / `adapters/` / `benchmarks/` | 预留给后续共享基础设施 |

## `src/diffaudit/attacks/`

| 文件 | 作用 |
| --- | --- |
| `secmi.py` / `secmi_adapter.py` | `SecMI` 的计划层、资产解析、dry-run、runtime probe、synthetic smoke |
| `pia.py` / `pia_adapter.py` | `PIA` 的计划层、资产解析、dry-run、runtime probe、runtime smoke、synthetic smoke |
| `clid.py` | `CLiD` 的计划层、asset probe、dry-run、dry-run smoke、artifact summary |
| `recon.py` | reconstruction-based 纯黑盒主线的计划层、asset probe、dry-run、eval smoke |
| `variation.py` | API-only variation 黑盒线的计划层、asset probe、dry-run、synthetic smoke |
| `registry.py` | 统一 planner 注册表 |

## 文档入口建议

如果你刚接手仓库，建议按这个顺序看：

1. [README.md](../README.md)
2. [docs/teammate-setup.md](teammate-setup.md)
3. [docs/data-and-assets-handoff.md](data-and-assets-handoff.md)
4. [docs/command-reference.md](command-reference.md)
5. [docs/reproduction-status.md](reproduction-status.md)
6. [workspaces/black-box/plan.md](../workspaces/black-box/plan.md)
7. [workspaces/gray-box/plan.md](../workspaces/gray-box/plan.md)
8. [workspaces/white-box/plan.md](../workspaces/white-box/plan.md)
9. [configs/README.md](../configs/README.md)
10. [src/diffaudit/README.md](../src/diffaudit/README.md)
