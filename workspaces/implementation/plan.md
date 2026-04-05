# 工程实现方向初始计划

## 状态面板

- `owner`: Codex
- `scope`: 共享 CLI、配置结构、dry-run、probe、结果 schema、参考资料索引
- `status`: 进行中，PIA runtime/smoke 已纳入最小复现线
- `blocked by`: 真实资产未到位、白盒/灰盒 threat model 仍在收敛
- `next command`: `conda run -n diffaudit-research python -m unittest tests.test_attack_registry tests.test_pia_adapter tests.test_smoke_pipeline`
- `last updated`: 2026-04-05

## 方向定位

工程方向负责为黑盒、白盒、灰盒三条线提供共享基础设施。

## 当前优先事项

1. 稳定 `SecMI` 执行层
2. 打通 `PIA` 的 runtime probe 与 synthetic smoke
3. 统一 config 结构
4. 统一 `ready / blocked / dry-run` 输出结构
5. 统一实验结果 schema

## 第一周目标

1. 补更深的 runtime probe
2. 明确资产契约
3. 减少对 ad hoc 命令的依赖
4. 让实验状态更容易被脚本和人共同理解
5. 给工作区、参考资料和配置补统一清单与模板
